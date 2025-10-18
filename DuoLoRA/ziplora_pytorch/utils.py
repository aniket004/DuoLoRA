import os
from typing import Optional, Dict
from huggingface_hub import hf_hub_download

import torch
from safetensors import safe_open
from diffusers import UNet2DConditionModel
from diffusers.loaders.lora import LORA_WEIGHT_NAME_SAFE
from .ziplora import ZipLoRALinearLayer, ZipLoRALinearLayerInference
import matplotlib.pyplot as plt


def get_lora_weights(
    lora_name_or_path: str, subfolder: Optional[str] = None, **kwargs
) -> Dict[str, torch.Tensor]:
    """
    Args:
        lora_name_or_path (str): huggingface repo id or folder path of lora weights
        subfolder (Optional[str], optional): sub folder. Defaults to None.
    """
    if os.path.exists(lora_name_or_path):
        if subfolder is not None:
            lora_name_or_path = os.path.join(lora_name_or_path, subfolder)
        if os.path.isdir(lora_name_or_path):
            lora_name_or_path = os.path.join(lora_name_or_path, LORA_WEIGHT_NAME_SAFE)
    else:
        lora_name_or_path = hf_hub_download(
            repo_id=lora_name_or_path,
            filename=LORA_WEIGHT_NAME_SAFE,
            subfolder=subfolder,
            **kwargs,
        )
    assert lora_name_or_path.endswith(
        ".safetensors"
    ), "Currently only safetensors is supported"
    tensors = {}
    with safe_open(lora_name_or_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def merge_lora_weights(
    tensors: torch.Tensor, key: str, prefix: str = "unet.unet."
) -> Dict[str, torch.Tensor]:
    """
    Args:
        tensors (torch.Tensor): state dict of lora weights
        key (str): target attn layer's key
        prefix (str, optional): prefix for state dict. Defaults to "unet.unet.".
    """
    target_key = prefix + key
    out = {}
    out_split_up = {}
    out_split_down = {}
    for part in ["to_q", "to_k", "to_v", "to_out.0"]:
        down_key = target_key + f".{part}.lora.down.weight"
        up_key = target_key + f".{part}.lora.up.weight"
        #print(f"Up: {tensors[up_key].shape}, Down: {tensors[down_key].shape}")
        merged_weight = tensors[up_key] @ tensors[down_key]
        out_split_up[part] = tensors[up_key]
        out_split_down[part] = tensors[down_key]
        #breakpoint()
        out[part] = merged_weight
    #breakpoint()
    return out, out_split_up, out_split_down

def initialize_ziplora_layer(state_dict_up, state_dict_down, state_dict_2_up, state_dict_2_down, part, **model_kwargs):
    ziplora_layer = ZipLoRALinearLayer(**model_kwargs)
    ziplora_layer.load_state_dict(
        {
            "up_1": state_dict_up[part],
            "up_2": state_dict_2_up[part],
            "down_1": state_dict_down[part],
            "down_2": state_dict_2_down[part],
            
        },
        strict=False,
    )
    return ziplora_layer


def unet_ziplora_state_dict(
    unet: UNet2DConditionModel, quick_release: bool = False
) -> Dict[str, torch.Tensor]:
    r"""
    Returns:
        A state dict containing just the LoRA parameters.
    """
    lora_state_dict = {}

    for name, module in unet.named_modules():
        if hasattr(module, "set_lora_layer"):
            lora_layer = getattr(module, "lora_layer")
            if lora_layer is not None:
                assert hasattr(lora_layer, "get_ziplora_weight"), lora_layer
                #breakpoint()
                weight = lora_layer.get_ziplora_weight()
                lora_state_dict[f"unet.{name}.lora.weight"] = weight
                if hasattr(lora_layer, "get_ziplora_weight1"):
                    content = lora_layer.get_ziplora_weight1()
                    style = lora_layer.get_ziplora_weight2()
                    lora_state_dict[f"unet.{name}.lora.content"] = content
                    lora_state_dict[f"unet.{name}.lora.style"] = style
                if quick_release:
                    lora_layer.cpu()
    return lora_state_dict

def ziplora_set_forward_type(unet: UNet2DConditionModel, type: str = "merge"):
    assert type in ["merge", "weight_1", "weight_2"]

    for name, module in unet.named_modules():
        if hasattr(module, "set_lora_layer"):
            lora_layer = getattr(module, "lora_layer")
            if lora_layer is not None:
                assert hasattr(lora_layer, "set_forward_type"), lora_layer
                lora_layer.set_forward_type(type)
    return unet


def ziplora_compute_mergers_similarity(unet):
    similarities = []
    for name, module in unet.named_modules():
        if hasattr(module, "set_lora_layer"):
            lora_layer = getattr(module, "lora_layer")
            if lora_layer is not None:
                assert hasattr(lora_layer, "compute_mergers_similarity"), lora_layer
                #breakpoint()
                similarities.append(lora_layer.compute_mergers_similarity())
    #similarity = torch.stack(similarities).sum(dim=0)
    similarity = torch.stack(similarities).mean()
    return similarity


## modified with layer priors
def ziplora_compute_mergers_rank_loss_cycle(unet, nuclear_norm_coeff, sparsity_coeff):
    rank_loss_per_layer = []
    for name, module in unet.named_modules():
        if hasattr(module, "set_lora_layer"):
            lora_layer = getattr(module, "lora_layer")
            if lora_layer is not None:
                # conditions for layers 15-45:
                #breakpoint()
                # if low res -> content:, elif mid_res -> ignore, else downblock, style only
                if  '.'.join(name.split('.')[:2]) in ['up_blocks.2', 'down_blocks.2', 'mid_blocks.2' ]: # 16x16
                    # R(content) > R(style)
                    rank_loss_per_layer.append(lora_layer.merger_rank_minimization_content(nuclear_norm_coeff, sparsity_coeff))
                elif '.'.join(name.split('.')[:2]) in ['up_blocks.1', 'down_blocks.1']: # 32x32
                    # R(style) > R(content)
                    rank_loss_per_layer.append(lora_layer.merger_rank_minimization_style(nuclear_norm_coeff, sparsity_coeff))
                      
    rank_loss = torch.stack(rank_loss_per_layer).mean()
    return rank_loss

def merge_lora_weights_for_inference(
    tensors: Dict[str, torch.Tensor], key: str, prefix: str = "unet.unet."
) -> Dict[str, torch.Tensor]:
    """
    Args:
        tensors (torch.Tensor): state dict of lora weights
        key (str): target attn layer's key
        prefix (str, optional): prefix for state dict. Defaults to "unet.unet.".
    """
    target_key = prefix + key
    out = {}
    for part in ["to_q", "to_k", "to_v", "to_out.0"]:
        key = target_key + f".{part}.lora.weight"
        out[part] = tensors[key]
    return out


def initialize_ziplora_layer_for_inference(state_dict, part, **model_kwargs):
    ziplora_layer = ZipLoRALinearLayerInference(**model_kwargs)
    ziplora_layer.load_state_dict(
        {
            "weight": state_dict[part],
        },
        strict=False,
    )
    return ziplora_layer

def fast_rank_lu_gpu(A, tol=1e-7):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    A = A.to(device)
    
    # LU decomposition
    LU, pivot = torch.linalg.lu_factor(A)
    rank = torch.sum(torch.abs(torch.diag(LU)) > tol)
    return rank.item()


def compute_lora_rank_beforemerge(tensors: Dict[str, torch.Tensor], key: str, prefix: str = "unet.unet."):
    """
    Args:
        tensors (torch.Tensor): state dict of lora weights
        key (str): target attn layer's key
        prefix (str, optional): prefix for state dict. Defaults to "unet.unet.".
    """
    target_key = prefix + key
    rank = {}
    for part in ["to_q", "to_k", "to_v", "to_out.0"]:
        key_c = target_key + f".{part}.lora.content"
        key_s = target_key + f".{part}.lora.style"
        #print('content', tensors[key_c])
        #print('style', tensors[key_s] )
        #breakpoint()
        rank[part] = (fast_rank_lu_gpu(tensors[key_c]), fast_rank_lu_gpu(tensors[key_s]))
 
    return rank


def insert_ziplora_to_unet(
    unet: UNet2DConditionModel, ziplora_name_or_path: str, **kwargs
):
    tensors = get_lora_weights(ziplora_name_or_path, **kwargs)
    #breakpoint()
    test_num = 0
    for attn_processor_name, attn_processor in unet.attn_processors.items():
        # Parse the attention module.
        attn_module = unet
        for n in attn_processor_name.split(".")[:-1]:
            attn_module = getattr(attn_module, n)
        # Get prepared for ziplora
        attn_name = ".".join(attn_processor_name.split(".")[:-1])
        state_dict = merge_lora_weights_for_inference(tensors, key=attn_name)
        # Set the `lora_layer` attribute of the attention-related matrices.
        kwargs = {"state_dict": state_dict}

        attn_module.to_q.set_lora_layer(
            initialize_ziplora_layer_for_inference(
                in_features=attn_module.to_q.in_features,
                out_features=attn_module.to_q.out_features,
                part="to_q",
                **kwargs,
            )
        )
        attn_module.to_k.set_lora_layer(
            initialize_ziplora_layer_for_inference(
                in_features=attn_module.to_k.in_features,
                out_features=attn_module.to_k.out_features,
                part="to_k",
                **kwargs,
            )
        )
        attn_module.to_v.set_lora_layer(
            initialize_ziplora_layer_for_inference(
                in_features=attn_module.to_v.in_features,
                out_features=attn_module.to_v.out_features,
                part="to_v",
                **kwargs,
            )
        )
        attn_module.to_out[0].set_lora_layer(
            initialize_ziplora_layer_for_inference(
                in_features=attn_module.to_out[0].in_features,
                out_features=attn_module.to_out[0].out_features,
                part="to_out.0",
                **kwargs,
            )
        )
        
    return unet


def insert_ziplora_to_unet_with_rank(
    unet: UNet2DConditionModel, ziplora_name_or_path: str, **kwargs
):
    tensors = get_lora_weights(ziplora_name_or_path, **kwargs)
    
    rank_content = {}
    rank_style = {}
   
    #breakpoint()
    for attn_processor_name, attn_processor in unet.attn_processors.items():
        # Parse the attention module.
        attn_module = unet
        for n in attn_processor_name.split(".")[:-1]:
            attn_module = getattr(attn_module, n)
        # Get prepared for ziplora
        attn_name = ".".join(attn_processor_name.split(".")[:-1])
        state_dict = merge_lora_weights_for_inference(tensors, key=attn_name)
        # Set the `lora_layer` attribute of the attention-related matrices.
        kwargs = {"state_dict": state_dict}

        attn_module.to_q.set_lora_layer(
            initialize_ziplora_layer_for_inference(
                in_features=attn_module.to_q.in_features,
                out_features=attn_module.to_q.out_features,
                part="to_q",
                **kwargs,
            )
        )
        attn_module.to_k.set_lora_layer(
            initialize_ziplora_layer_for_inference(
                in_features=attn_module.to_k.in_features,
                out_features=attn_module.to_k.out_features,
                part="to_k",
                **kwargs,
            )
        )
        attn_module.to_v.set_lora_layer(
            initialize_ziplora_layer_for_inference(
                in_features=attn_module.to_v.in_features,
                out_features=attn_module.to_v.out_features,
                part="to_v",
                **kwargs,
            )
        )
        attn_module.to_out[0].set_lora_layer(
            initialize_ziplora_layer_for_inference(
                in_features=attn_module.to_out[0].in_features,
                out_features=attn_module.to_out[0].out_features,
                part="to_out.0",
                **kwargs,
            )
        )
        
        ranks = compute_lora_rank_beforemerge(tensors, attn_name)
        print(f"rank of layer {attn_processor_name} of to_out is content: {ranks['to_out.0'][0]} and style: content: {ranks['to_out.0'][1]}") 
        print(f"rank of layer {attn_processor_name} of to_k is is content: {ranks['to_k'][0]} and style: content: {ranks['to_k'][1]}")
        print(f"rank of layer {attn_processor_name} of to_v is is content: {ranks['to_v'][0]} and style: content: {ranks['to_v'][1]}")
        print(f"rank of layer {attn_processor_name} of to_q is is content: {ranks['to_q'][0]} and style: content: {ranks['to_q'][1]}")
        
    
        if attn_processor_name not in rank_content:
            rank_content[attn_processor_name] = {}
            rank_style[attn_processor_name] = {}

        for part in ranks:
            rank_content[attn_processor_name][part] = ranks[part][0]
            rank_style[attn_processor_name][part] = ranks[part][1]

    return unet, rank_content, rank_style

def filter_and_plot_histogram(data_content, data_style, name):
    
    filtered_vals_low_content = [v for k, v in data_content.items() if '.'.join(name.split('.')[:2]) in ['up_blocks.2', 'down_blocks.2', 'mid_blocks.2' ]] # 16x16
    filtered_vals_low_style = [v for k, v in data_style.items() if '.'.join(name.split('.')[:2]) in ['up_blocks.2', 'down_blocks.2', 'mid_blocks.2' ]] # 16x16

    
    filtered_vals_high_content = [v for k, v in data_content.items() if '.'.join(name.split('.')[:2]) in ['up_blocks.1', 'down_blocks.1']] # 32x32
    filtered_vals_high_style = [v for k, v in data_style.items() if '.'.join(name.split('.')[:2]) in ['up_blocks.1', 'down_blocks.1']] # 32x32


    plt.hist(filtered_vals_low_content, bins=5, edgecolor='black', alpha=0.7)
    plt.title("Histogram of low resolution values in Content")
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.savefig('hist_low_content')
    plt.close()
    
    plt.hist(filtered_vals_high_content, bins=5, edgecolor='black', alpha=0.7)
    plt.title("Histogram of high resolution values in Content")
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.savefig('hist_high_content')
    plt.close()
    
    plt.hist(filtered_vals_low_style, bins=5, edgecolor='black', alpha=0.7)
    plt.title("Histogram of low resolution values in Style")
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.savefig('hist_low_style')
    plt.close()
    
    plt.hist(filtered_vals_high_style, bins=5, edgecolor='black', alpha=0.7)
    plt.title("Histogram of high resolution values in Style")
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.savefig('hist_high_style')
    plt.close()
    