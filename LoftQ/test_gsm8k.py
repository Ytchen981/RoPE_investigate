#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import logging
import math
import re
import os
from dataclasses import dataclass, field
from typing import Dict, Optional
import json

import torch
import numpy as np
import transformers
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from transformers import Trainer, AutoModel

from peft import PeftModel
from datasets import load_dataset
from accelerate.utils import set_seed


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
        default="LoftQ/Mistral-7B-v0.1-4bit-64rank",
        metadata={"help": "Path to the model."},
    )
    adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the LoRA adapter. Used in evaluation or resuming from the checkpoint."},
    )
    ckpt_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to your local output directory"},
    )
    full_precision:  bool = field(default=False)
    token: Optional[str] = field(
        default=None,
        metadata={"help": "HF token to access to private models, e.g., meta-llama"},
    )
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be left padded (and possibly truncated)."},
    )
    adjust_ratio: float = field(
        default=0.,
        metadata={"help": "Adjust ratio between orthogonal and non-orthogonal channels"
                  },
    )
    mask_threshold: float = field(
        default=0.1,
        metadata={"help": "Threshold for distinguishing orthogonal pair and non-orthogonal pair."}
    )


@dataclass
class DataArguments:
    data_name: str = field(default="gsm8k", metadata={"help": "Dataset name."})
    batch_size: int = field(default=16, metadata={"help": "Evaluation batch size."})


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

def generate_forward_hook(weight, n_heads, hf_style=True, ratio=0.1, threshold=0.1, name=None):
    dim = weight.shape[0]
    if hf_style:
        w1_list = []
        w2_list = []
        head_dim = dim // n_heads
        for i in range(n_heads):
            w1_list.append(weight[i * head_dim:(i * head_dim + head_dim // 2), :])
            w2_list.append(weight[(i * head_dim + head_dim // 2): (i + 1) * head_dim, :])
        #W1 = torch.cat(w1_list, dim=0)
        #W2 = torch.cat(w2_list, dim=0)
        # W1 = weight[:dim // 2]
        # W2 = weight[dim // 2:]
    else:
        W1 = weight[::2]
        W2 = weight[1::2]
        half_head_dim = dim // (2 * n_heads)
        w1_list = [W1[i * half_head_dim:(i + 1) * half_head_dim, :] for i in range(n_heads)]
        w2_list = [W2[i * half_head_dim:(i + 1) * half_head_dim, :] for i in range(n_heads)]
    '''mask_list = []
    for i in range(n_heads):
        abs_cos = F.cosine_similarity(w1_list[i], w2_list[i], dim=1).abs()
        pair_norm = torch.sqrt(torch.sum(w1_list[i]**2, dim=1) + torch.sum(w2_list[i]**2, dim=1))
        corr_coef = np.corrcoef(abs_cos.cpu(), pair_norm.cpu())[0, 1]
        min_abs_cos = abs_cos.min().item()
        max_abs_cos = abs_cos.max().item()
        current_head_mask = (abs_cos - min_abs_cos) / (max_abs_cos - min_abs_cos) * corr_coef
        mask_list.append(current_head_mask)
    mask = torch.cat(mask_list, dim=0)'''
    cos_list = []
    norm_list = []
    for i in range(n_heads):
        abs_cos = F.cosine_similarity(w1_list[i], w2_list[i], dim=1).abs()
        pair_norm = torch.sqrt(torch.sum(w1_list[i] ** 2, dim=1) + torch.sum(w2_list[i] ** 2, dim=1))
        cos_list.append(abs_cos)
        norm_list.append(pair_norm)
    cos_total = torch.cat(cos_list, dim=0)
    norm_total = torch.cat(norm_list, dim=0)
    corr_coef = np.corrcoef(cos_total.cpu(), norm_total.cpu())[0, 1]
    mask = (cos_total > threshold).int().squeeze()

    if hf_style:
        #extended_mask = mask.repeat(2).flatten()
        mask = mask.view(n_heads, (dim // (2 * n_heads)))
        extended_mask = mask.repeat(1, 2).flatten()
    else:
        extended_mask = mask.repeat(2, 1)
        extended_mask = extended_mask.transpose().contiguous().flatten()


    extended_mask = extended_mask.unsqueeze(0).unsqueeze(0).cuda()
    extended_mask = extended_mask * ratio * corr_coef + 1
    #print(extended_mask)
    #exit()

    def adjust_weight_forward_hook(module, input, output):
        assert isinstance(module, nn.Linear)
        if isinstance(output, tuple):
            origin_output = output[0]

            target_output = origin_output * extended_mask.to(origin_output.dtype)
            return (target_output,)
        else:
            target_output = output * extended_mask.to(output.dtype)
            return target_output

    return adjust_weight_forward_hook, mask

def register_mask_hook(model, base_model, ratio=0.1, threshold=0.1, hf_style=True):
    q_name = "q_proj"
    k_name = "k_proj"
    #prefix = "lora_B"
    handle_dict = {}
    origin_model = AutoModel.from_pretrained(base_model, trust_remote_code=True)
    with open(os.path.join(base_model, "config.json")) as f:
        params = json.load(f)
    n_heads = params["num_attention_heads"]
    for n, m in tqdm(model.named_modules(), desc="Register forward hook"):
        if isinstance(m, nn.Linear) and (n.split(".")[-1] == q_name):
            if "base_model" in n:
                weight_name = '.'.join(n.split('.')[3:])+".weight"
            else:
                weight_name = '.'.join(n.split('.')[1:]) + ".weight"
            weight = origin_model.state_dict()[weight_name]
            hook_func, mask = generate_forward_hook(weight, n_heads, hf_style, ratio, threshold, n)
            handle_dict[n] = m.register_forward_hook(hook_func)
            #mask = mask.flatten().to(m.weight.data.device)
            #print(m.weight.data.shape)
            #m.weight.data *= mask.unsqueeze(1)
            #if hasattr(m, "bias") and m.bias is not None:
                #m.bias.data *= mask
    return handle_dict

def remove_hooks(handle_dict):
    for key in handle_dict.keys():
        handle_dict[key].remove()

def evaluation(model_args, data_args):
    if model_args.full_precision:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            token=model_args.token,
            device_map='auto',
        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            token=model_args.token,
            device_map='auto',
            quantization_config=transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type='nf4',
            ),
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        token=model_args.token,
        model_max_length=model_args.model_max_length,
        padding_side="left",
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

    ##########################
    #       Peft Model       #
    ##########################
    if model_args.adapter_name_or_path is not None:
        model = PeftModel.from_pretrained(
            model,
            model_args.adapter_name_or_path,
            is_trainable=False,
            token=model_args.token,
        )
        model = model.to('cuda')
    else:
        '''model = PeftModel.from_pretrained(
            model,
            model_args.model_name_or_path,
            subfolder='gsm8k',
            is_trainable=False,
            token=model_args.token,
        )'''
        pass
    #model = model.to('cuda')
    '''handle_dict = register_mask_hook(model, model_args.model_name_or_path,
                                     ratio=model_args.adjust_ratio, threshold=model_args.mask_threshold,
                                     hf_style=True)'''

    ######################
    #      dataset       #
    ######################
    logging.warning("Downloading Data")
    dataset = load_dataset(data_args.data_name, "main")
    test_set = dataset['test']

    logging.warning("Formatting inputs...")
    question = [f"{example['question']}{QUESTION_PROMPT}" for example in test_set]
    answer = []

    # get numerical answer
    for example in test_set['answer']:
        ans = example.split('####')[-1]
        ans = ans.replace(',', '')  # handle numbers like 2,000
        try:
            ans = float(ans)
        except ValueError:
            ans = float("inf")
        answer.append(ans)

    logging.warning("Tokenizing inputs...")
    eval_step = math.ceil(len(question)/data_args.batch_size)
    logging.warning(f"Total example: {len(question)} | eval batch size: {data_args.batch_size}"
                    f"eval steps: {eval_step}")
    question_data = []
    for i in range(eval_step):
        if i < eval_step - 1:
            batch = tokenizer(
                question[i*data_args.batch_size: (i+1)*data_args.batch_size],
                return_tensors="pt",
                padding="longest",
            )
        else:
            batch = tokenizer(
                question[i*data_args.batch_size:],
                return_tensors="pt",
                padding="longest",
            )
        batch['input_len'] = len(batch['input_ids'][0])
        question_data.append(batch)

    model.eval()
    gen_kwargs = {
        "max_new_tokens": 256,
        "temperature": 0.1,
        "top_k": 40,
        "top_p": 0.95,
        "do_sample": True,
    }
    ans_pred_list = []
    set_seed(42)
    for step, batch in enumerate(question_data):
        with torch.no_grad():
            gen_kwargs["input_ids"] = batch["input_ids"].to('cuda')
            gen_kwargs["attention_mask"] = batch["attention_mask"].to('cuda')
            generated_tokens = model.generate(**gen_kwargs)

        pred_tokens = generated_tokens[:, batch['input_len']:]
        decoded_pred = tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)

        # Extract the numbers in sentences
        print(decoded_pred)
        ans_pred_list += [extract_answer_number(sentence_pred) for sentence_pred in decoded_pred]

    print("prediction", ans_pred_list)
    print("ground truth", answer)

    accuracy = compute_accuracy(answer, ans_pred_list)

    print(f"adapter: {model_args.adapter_name_or_path} | GSM8K test accuracy: {100*accuracy:.2f}% | "
          f"full precision: {model_args.full_precision} | adjust ratio:{model_args.adjust_ratio} | threshold: {model_args.mask_threshold}")
    if model_args.adapter_name_or_path is None:
        result_file = os.path.join(model_args.model_name_or_path, "accuracy_eval_v3.txt")
    else:
        result_file = os.path.join(model_args.adapter_name_or_path, "accuracy_eval_v3.txt")
    with open(result_file, "a+") as f:
        f.write(f"new v3 adapter: {model_args.adapter_name_or_path} | GSM8K test accuracy: {100*accuracy:.2f}% | "
          f"full precision: {model_args.full_precision}| adjust ratio:{model_args.adjust_ratio} | threshold: {model_args.mask_threshold}\n")

    #remove_hooks(handle_dict)

def extract_answer_number(sentence: str) -> float:
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        return float('inf')
    segment = sentence.split(ANSWER_PROMPT)
    if len(segment) > 1:
        pred_answer = segment[1]
        pred_answer = [s for s in re.findall(r'-?\d+\.?\d*', pred_answer)]
        if len(pred_answer) > 0:
            pred_answer = pred_answer[0]
        else:
            pred_answer = float(pred[-1])
    else:
        # use the last number as the answer
        pred_answer = float(pred[-1])

    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float('inf')
    return pred_answer


def compute_accuracy(pred: list, gold: list):
    acc = 0.0
    for p, g in zip(pred, gold):
        if p == g:
            acc += 1

    return acc / len(pred)


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    if model_args.ckpt_dir is not None:
        adapter_dir_list = [os.path.join(model_args.ckpt_dir, ckpt_dir) for ckpt_dir in os.listdir(model_args.ckpt_dir)
                            if 'checkpoint-' in ckpt_dir]
    elif model_args.adapter_name_or_path is not None:
        adapter_dir_list = [model_args.adapter_name_or_path]
    else:
        logging.warning("Use the checkpoint in HF hub, stored in the `subfolder='gsm8k'` in target model.")
        adapter_dir_list = [None]

    for adapter_path in adapter_dir_list:
        model_args.adapter_name_or_path = adapter_path
        evaluation(model_args, data_args)
