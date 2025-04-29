#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import os
import copy
from typing import Dict, cast, IO, List, Optional
from pathlib import Path

import torch
import torch.nn as nn
from groundingdino.models.GroundingDINO.modules.lora import LoRALayer, Linear
from groundingdino.models.GroundingDINO.modules.lora_pool import LinearPool
from groundingdino.models.GroundingDINO.modules.multi_head_attention import MultiHeadAttention

from detectron2.checkpoint import DetectionCheckpointer
from groundingdino.util.task_memory import TaskMemory


class DetectionLoraCheckpointer(DetectionCheckpointer):
    def __init__(self, model, save_dir="", *, save_to_disk=None, args, **checkpointables):
        self.args = args
        super().__init__(
            model,
            save_dir,
            save_to_disk=save_to_disk,
            **checkpointables,
        )

    def save(self, name, **kwargs):
        """
        Dump model and checkpointables to a file.

        Args:
            name (str): name of the file.
            kwargs (dict): extra arbitrary data to save.
        """
        if not self.save_dir or not self.save_to_disk:
            return

        data = {"model": lora_state_dict(self.model), "args": self.args}
        assert data["model"]
        for key, obj in self.checkpointables.items():
            data[key] = obj.state_dict()
        data.update(kwargs)

        basename = f"{name}.pth"
        save_file = os.path.join(self.save_dir, basename)
        assert os.path.basename(save_file) == basename, basename
        with self.path_manager.open(Path(self.save_dir).parent.parent / "last_lora.pth", "wb") as f:
            torch.save(data, cast(IO[bytes], f))
        self.tag_last_checkpoint(basename)

def get_sorted_directories(output_dir: str) -> List[str]:
    
    def get_lora_mtime(dir_name: str) -> Optional[float]:
        lora_path = os.path.join(output_dir, dir_name, 'lora_final.pth')
        if os.path.isfile(lora_path):
            return os.path.getmtime(lora_path)
        return None
    
    directories = [
        d for d in os.listdir(output_dir) 
        if os.path.isdir(os.path.join(output_dir, d)) and 
        get_lora_mtime(d) is not None
    ]
    
    sorted_directories = sorted(
        directories,
        key=lambda d: get_lora_mtime(d)
    )
    
    return sorted_directories

@torch.no_grad()
def merge_lora(model, output_dir, alpha=None, must_exist=False, scale=None):
    checkpoints = []
    output_dir = f'{output_dir}/odinw13/'
    directories = get_sorted_directories(output_dir)

    alpha = 1.0 if alpha is None else alpha 

    for dataset_dir in directories:
        checkpoint_path = os.path.join(output_dir, dataset_dir, 'lora_final.pth')
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location = 'cpu')
            checkpoints.append(checkpoint)

    num_tasks = len(checkpoints)
    if not checkpoints:
        assert not must_exist, f'No tasks found in {output_dir}'
        return

    original_state_dict = copy.deepcopy(model.state_dict())
    model_parameters = model.state_dict()
    lora_keys = checkpoints[0]['model'].keys()
    for key in lora_keys:
        if not key.endswith('.lora_A'):
            continue

        layer_name = key.rsplit('.', 1)[0]
        model_param = model_parameters[f'{layer_name}.weight'].type(torch.float64) 
        product_sum = torch.zeros_like(model_param)
        
        for _, checkpoint in enumerate(checkpoints):
            A = checkpoint['model'][f'{layer_name}.lora_A'].type(torch.float64)
            B = checkpoint['model'][f'{layer_name}.lora_B'].type(torch.float64)
            product_sum += 1 / num_tasks * (B @ A)

        model_param += alpha * product_sum 
        model_parameters[f'{layer_name}.weight'] = model_param.type(torch.float32) 

    model.load_state_dict(model_parameters)

    # assert all parameters have changed
    same_params = []
    for key in lora_keys:
        key = key.rsplit('.', 1)[0] + '.weight'
        p1 = original_state_dict[key]
        p2 = model.state_dict()[key]
        if torch.equal(p1, p2):
            same_params.append(key)
    if same_params:
        print(f'[WARNING!] The following parameters have not been changed: {same_params}')

    print('Lora has been merged')

def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and \
                hasattr(m, 'bias') and \
                m.bias is not None:
                    m.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_')[0]+'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError
    

@torch.no_grad()
def apply_custom_attention(module, attention_module_names):
    assert attention_module_names is not None

    for attn_module_name in attention_module_names:
        original_self_attn = copy.deepcopy(getattr(module, attn_module_name))

        setattr(module, attn_module_name, MultiHeadAttention(original_self_attn.embed_dim, original_self_attn.num_heads))
        # clone weights from original self_attn
        qkv_weight = original_self_attn.in_proj_weight.clone()
        qkv_bias = original_self_attn.in_proj_bias.clone()
        o_w = original_self_attn.out_proj.weight.clone()
        o_b = original_self_attn.out_proj.bias.clone()
        # chunk qkv_weight and qkv_bias
        q_w, k_w, v_w = torch.chunk(qkv_weight, 3, dim=0)
        q_b, k_b, v_b = torch.chunk(qkv_bias, 3, dim=0)
        # assign weights and biases to new self_attn
        attn_module = getattr(module, attn_module_name)
        attn_module.linear_q.weight = torch.nn.Parameter(q_w)
        attn_module.linear_k.weight = torch.nn.Parameter(k_w)
        attn_module.linear_v.weight = torch.nn.Parameter(v_w)
        attn_module.linear_o.weight = torch.nn.Parameter(o_w)
        attn_module.linear_q.bias = torch.nn.Parameter(q_b)
        attn_module.linear_k.bias = torch.nn.Parameter(k_b)
        attn_module.linear_v.bias = torch.nn.Parameter(v_b)
        attn_module.linear_o.bias = torch.nn.Parameter(o_b)


def get_lora_modules(gd, pool_out_min=128, exclude_layers=['transformer.dec', 'bert', 'backbone', 'feat_map']):
    if exclude_layers is None:
        exclude_layers = []

    return_modules = []

    for m_name, module in dict(gd.named_modules()).items():
        for l_name, layer in dict(module.named_children()).items():
            if not isinstance(layer, nn.Linear) or layer.out_features < pool_out_min:
                continue

            id_ = f'{m_name}.{l_name}' if m_name else l_name
            if exclude_layers and any(id_.startswith(exclude_layer) for exclude_layer in exclude_layers):
                continue

            if not layer.weight.requires_grad:
                continue

            return_modules.append(
                {
                    'id': id_,
                    'module': module,
                    'm_name': m_name,
                    'l_name': l_name,
                    'layer': layer,
                }
            )
    return return_modules

def apply_lora(modules, r=8, lora_alpha=8, lora_dropout=0.0, scaling=None, lambda_a_marouf=0.3, lambda_b_marouf=0.7):
    total_in_features = 0
    total_out_features = 0
    total_in_gt_out = 0
    total_out_gt_in = 0
    for m in modules:
        m_id = m['id']
        module = m['module']
        l_name = m['l_name']
        layer = m['layer']

        all_classes = {class_name[0].lower() + class_name[1:] for dataset_name, dataset_classes in TaskMemory().task_mapping.items() for class_name in dataset_classes.values() if dataset_name != 'coco_2017_val'}
        lora_layer = LinearPool(
            in_features=layer.in_features,
            out_features=layer.out_features,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            scaling=scaling,
            layer_name=m_id,
            classes=all_classes,
            lambda_a_marouf=lambda_a_marouf,
            lambda_b_marouf=lambda_b_marouf,
            )
        
        with torch.no_grad():
            lora_layer.weight = torch.nn.Parameter(layer.weight.clone())
            lora_layer.bias = torch.nn.Parameter(layer.bias.clone()) if layer.bias is not None else None
        setattr(module, l_name, lora_layer.to(layer.weight.device))
        total_in_features += layer.in_features
        total_out_features += layer.out_features
        if layer.in_features > layer.out_features:
            total_in_gt_out += 1
        elif layer.out_features > layer.in_features:
            total_out_gt_in += 1