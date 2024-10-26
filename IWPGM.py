import os
import numpy as np
import json
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel


def generate_backward_hook(weight, threshold, random, n_heads, hf_style=True, name=None):
    dim = weight.shape[0]
    if hf_style: # Get the weight vector pairs based on RoPE implementation, hf_style refers to the RoPE implementation in models on huggingface.
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
    abs_cos = F.cosine_similarity(W1, W2, dim=1).abs() # Calculate the absolute cosine similarity between each pair of weight vectors.
    mask = (abs_cos < threshold) # Generate the mask based on the threshold
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
    def mask_backward_hook(module, grad_output): # Define the backward hook applying the mask on the gradient.
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
            parent = '.'.join(n.split('.')[3:-2])+".weight" #locate the target weight, could be changed for different models
            weight = origin_model.state_dict()[parent] #load the pretrained weights to determine the mask
            hook_func, mask = generate_backward_hook(weight, threshold, random, n_heads, hf_style, n) #generate the backward hook function
            handle_dict[n] = m.register_full_backward_pre_hook(hook_func) #register the hook
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