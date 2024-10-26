import os
import sys
from typing import List
import json

import fire
import torch
import transformers
from datasets import load_dataset
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModel, AutoTokenizer

from utils.prompter import Prompter


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
        #extended_mask = mask.repeat(2).flatten()
        mask = mask.view(n_heads, (dim // (2 * n_heads)))
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

def register_mask_hook(model, base_model, output_dir, threshold=0.1, random=False, hf_style=True):
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
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    with open(os.path.join(output_dir, "param_ratio.txt"), "a") as f:
        f.write(f"full trainable params: {origin_trainable_params} || "
        f"after mask trainable params: {mask_trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * mask_trainable_params / all_param} || "
        f"mask ratio: {100 * (1 - mask_trainable_params / origin_trainable_params):.2f}")
    plt.figure()
    plt.plot([i+2 for i in range(len(ratio_list))], ratio_list)
    plt.xlabel("layer")
    plt.ylabel("remain ratio")
    plt.savefig(os.path.join(output_dir, "ratio_layer.png"))
    return handle_dict

def remove_hooks(handle_dict):
    for key in handle_dict.keys():
        handle_dict[key].remove()

def train(
    # model/data params
    base_model: str = "",  # the only required argument
    data_path: str = "yahma/alpaca-cleaned",
    output_dir: str = "./lora-alpaca",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 2000,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
    # cos based mask param
    threhold: float = 0.1,
    random_mask: bool = False,
    hf_style: bool = True,
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    '''model = AutoModel.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )'''

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    #tokenizer = AutoTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    #model.print_trainable_parameters()  # Be more transparent about the % of trainable params.
    handle_dict = register_mask_hook(model, base_model, output_dir, threshold=threhold, random=random_mask, hf_style=hf_style)   # register mask hook and print trainable params

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    '''old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))'''

    '''if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)'''

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    remove_hooks(handle_dict)

    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    fire.Fire(train)
