# Modified from https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py

import copy
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import json

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer, AutoModel


from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from datasets import load_dataset


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
ANSWER_PROMPT = "The final answer is: "
QUESTION_PROMPT = "\nAnswer the above question. First think step by step and then answer the final number.\n"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        #default="LoftQ/Mistral-7B-v0.1-4bit-64rank",
        default="/home_new/chenyiting/RoPE_angle/model/Llama-2-7b-hf",
        metadata={"help": "Path to the model."},
    )
    adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the LoRA adapter. Used in evaluation or resuming from the checkpoint."},
    )
    lora_init: bool = field(
        default=False,
        metadata={"help": "True: Use zero and gaussian initialization; False: Load adapters from LoftQ in HF hub."},
    )
    full_precision:  bool = field(
        default=False,
        metadata={"help": "False: Use bitsandbytes Linear4bit, real quantization"
                          "True: Use quantization equivalent fp16/fp32 weights."
                  },
    )
    rank: int = field(
        default=64,
        metadata={"help": "Rank of LoRA adapters. LoftQ does not require this config. Used for fp16 LoRA or QLoRA."},
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoftQ does not require this config. Used for QLoRA."},
    )
    token: Optional[str] = field(
        default=None,
        metadata={"help": "HF token to access to private models, e.g., meta-llama"},
    )
    mask_threshold: float = field(
        default=0.,
        metadata={"help": "threshold for masked channel"},
    )
    random_mask: bool = field(
        default=False,
        metadata={"help": "Whether random mask should be used or not."
                  },
    )


@dataclass
class DataArguments:
    data_name: str = field(
        default="gsm8k",
        metadata={"help": "Dataset name."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    expt_name: str = field(
        default="default",
        metadata={"help": "Experiment name"},
    )


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(sources: Sequence[str], targets: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Preprocess the data by tokenizing."""
    # sources are questions, and targets are answers
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        logging.warning("Formatting inputs...")
        sources = [f"{example['question']}{QUESTION_PROMPT}" for example in raw_data]
        targets = [f"{example['answer']}{tokenizer.eos_token}".replace("####", ANSWER_PROMPT) for example in raw_data]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    logging.warning("Downloading Data")
    dataset = load_dataset(data_args.data_name, "main")
    train_set = dataset['train']
    train_dataset = SupervisedDataset(raw_data=train_set, tokenizer=tokenizer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

def generate_backward_hook(weight, threshold, random, n_heads, hf_style=True, name=None):
    dim = weight.shape[0]
    if hf_style:
        w1_list = []
        w2_list = []
        head_dim = dim // n_heads
        for i in range(n_heads):
            w1_list.append(weight[i*head_dim:(i*head_dim + head_dim//2),:])
            w2_list.append(weight[(i*head_dim+head_dim//2): (i+1)*head_dim,:])
        W1 = torch.cat(w1_list, dim=0)
        W2 = torch.cat(w2_list, dim=0)
        #W1 = weight[:dim // 2]
        #W2 = weight[dim // 2:]
    else:
        W1 = weight[::2]
        W2 = weight[1::2]
    abs_cos = F.cosine_similarity(W1, W2, dim=1).abs()
    mask = (abs_cos < threshold)
    mask = mask.int().squeeze()
    if random:
        unmask_num = mask.sum().item()
        np.random.seed(13)
        unmask_index = np.random.choice([i for i in range(dim // 2)], unmask_num, replace=False)
        mask = torch.zeros(dim // 2)
        mask[unmask_index] += 1
        assert mask.sum() == unmask_num

    if hf_style:
        mask = mask.view(n_heads, -1)
        extended_mask = mask.repeat(1, 2).flatten()
    else:
        extended_mask = mask.repeat(2, 1)
        extended_mask = extended_mask.transpose().contiguous().flatten()

    extended_mask = extended_mask.unsqueeze(0).unsqueeze(0).cuda()
    def mask_backward_hook(module, grad_output):
        assert isinstance(module, nn.Linear)

        weight_grad = grad_output[0]

        weight_grad *= extended_mask
        return (weight_grad,)

    return mask_backward_hook, extended_mask

def register_mask_hook(model, base_model, threshold=0.1, random=False, hf_style=True):
    q_name = "q_proj"
    k_name = "k_proj"
    prefix = "lora_B"
    handle_dict = {}
    num_dict = {}
    ratio_list = []
    origin_model = AutoModel.from_pretrained(base_model, trust_remote_code=True)
    with open(os.path.join(base_model, "config.json")) as f:
        params = json.load(f)
    n_heads = params["num_attention_heads"]
    for n, m in tqdm(model.named_modules(), desc="Register backward hook"):
        if isinstance(m, nn.Linear) and prefix in n and (q_name in n or k_name in n) and (int(n.split('.')[4])>=2):
            parent = '.'.join(n.split('.')[3:-2])+".weight"
            weight = origin_model.state_dict()[parent]
            hook_func, mask = generate_backward_hook(weight, threshold, random, n_heads, hf_style, n)
            handle_dict[n] = m.register_full_backward_pre_hook(hook_func)
            mask = mask.flatten().to(m.weight.data.device)
            print(m.weight.data.shape)
            m.weight.data *= mask.unsqueeze(1)
            if hasattr(m, "bias") and m.bias is not None:
                m.bias.data *= mask
            num_dict[n] = mask.sum()
            ratio_list.append(100 * mask.sum().item()/mask.numel())
    # print trainable parameters
    origin_trainable_params = 0
    mask_trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            origin_trainable_params += param.numel()
            Flag = True
            for key in num_dict.keys():
                if key in name:
                    mask_trainable_params += param.numel() // param.shape[0] * num_dict[key]
                    Flag = False
                    break
            if Flag:
                mask_trainable_params += param.numel()
    print(
        f"full trainable params: {origin_trainable_params} || "
        f"after mask trainable params: {mask_trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * mask_trainable_params / all_param} || "
        f"mask ratio: {100 * (1 - mask_trainable_params / origin_trainable_params):.2f}"
    )
    return handle_dict

def remove_hooks(handle_dict):
    for key in handle_dict.keys():
        handle_dict[key].remove()



def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.full_precision:
        if any(name in model_args.model_name_or_path.lower() for name in ["falcon"]):
            model = transformers.FalconForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16,
                token=model_args.token,
                trust_remote_code=True,
            )
        else:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16,
                token=model_args.token,
                trust_remote_code=True,
            )
    else:
        if any(name in model_args.model_name_or_path.lower() for name in ["falcon"]):
            model = transformers.FalconForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16,
                token=model_args.token,
                quantization_config=transformers.BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=False,
                    bnb_4bit_quant_type='nf4',
                ),
                trust_remote_code=True,
            )
        else:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16,
                token=model_args.token,
                quantization_config=transformers.BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=False,
                    bnb_4bit_quant_type='nf4',
                ),
                trust_remote_code=True,
            )

    ##########################
    #       Peft Model       #
    ##########################
    if model_args.lora_init:
        task_type = TaskType.CAUSAL_LM
        if any(name in model_args.model_name_or_path.lower() for name in ["llama", "mistral", "falcon"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
        elif any(name in model_args.model_name_or_path.lower() for name in ["phi"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
        else:
            raise ValueError(f"Only support LLAMA, Mistral, Falcon, Phi-2, but got {model_args.model_name_or_path}.")
        lora_config = LoraConfig(
            task_type=task_type,
            inference_mode=False,
            r=model_args.rank,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=0.1,
            target_modules=target_modules,
            init_lora_weights=True,
        )
        model = get_peft_model(model, lora_config)
    elif model_args.adapter_name_or_path is not None:
        model = PeftModel.from_pretrained(
            model,
            model_args.adapter_name_or_path,
            is_trainable=True,
            token=model_args.token,
        )
    else:
        model = PeftModel.from_pretrained(
            model,
            model_args.model_name_or_path,
            subfolder='loftq_init',
            is_trainable=True,
            token=model_args.token,
        )
    if model_args.mask_threshold > 0:
        handle_dict = register_mask_hook(model, model_args.model_name_or_path,  threshold=model_args.mask_threshold, random=model_args.random_mask,
                                         hf_style=True)
    else:
        handle_dict = None

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        token=model_args.token,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    training_args.output_dir = os.path.join(
        training_args.output_dir,
        training_args.expt_name,
        model_args.model_name_or_path.split('/')[-1],
        f"ep_{int(training_args.num_train_epochs)}",
        f"lr_{training_args.learning_rate}",
        f"seed_{training_args.seed}",
    )
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    if handle_dict is not None:
        remove_hooks(handle_dict)
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
