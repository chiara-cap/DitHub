from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, Set
from groundingdino.models.GroundingDINO.modules.lora import LoRALayer
from groundingdino.util.task_memory import TaskMemory

class LinearPool(nn.Linear, LoRALayer):
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        scaling: Optional[float] = None,
        layer_name: str = None,
        classes: Set[str] = None,
        lambda_a_marouf: float = 0.3,
        lambda_b_marouf: float = 0.7,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        task_memory = TaskMemory()
        
        assert layer_name, 'Please set an layer_name for this pool, so it can be retrieved by the TaskMemory'
        assert not fan_in_fan_out and r > 0, 'parameter not supported/tested in this version'
        self.fan_in_fan_out = fan_in_fan_out

        self.layer_name = layer_name
        self.per_class = False
        self.lambda_a_marouf = lambda_a_marouf
        self.lambda_b_marouf = lambda_b_marouf

        # Actual trainable parameters
        if r > 0:
            per_class_lora_A = {}

            self.shared_lora_b = nn.Parameter(self.weight.new_zeros((self.out_features, self.r)))
            nn.init.zeros_(self.shared_lora_b)

            for current_class in classes:
                per_class_lora_A[f'class_{current_class}'] = nn.Parameter(self.weight.new_zeros((self.r, self.in_features)))
                nn.init.kaiming_uniform_(per_class_lora_A[f'class_{current_class}'], a=math.sqrt(5))
            
            self.warmup_lora_a = nn.Parameter(self.weight.new_zeros((self.r, self.in_features)))
            nn.init.kaiming_uniform_(self.warmup_lora_a, a=math.sqrt(5))
            
            self.per_class_lora_A = nn.ParameterDict(per_class_lora_A)

            task_memory.set_counter(classes)

            self.scaling = scaling if scaling is not None else self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

            # load lora_A and lora_B for each class from previous tasks
            per_class_lora_A_and_B = task_memory.get_modules(layer_name)
            for class_name, ab_tensors in per_class_lora_A_and_B.items():
                # A
                self.per_class_lora_A[class_name] = ab_tensors[0]

            if self.training and task_memory.task_index > 0:
                self.shared_lora_b = nn.Parameter(task_memory.get_memory("B_shared", layer_name))
                task_memory.set_memory("B_prev_task", layer_name, self.shared_lora_b.data.detach().clone())

        nn.Linear.reset_parameters(self)
        task_memory.register_module(self)

    @torch.no_grad()
    def enable_per_class(self, current_task):
        task_memory = TaskMemory()
        classes = [x[0].lower() + x[1:] for x in task_memory.task_mapping[current_task.lower()].values()]
        if len(classes) == 1:
            if task_memory._class_counter[classes[0]] > 0:  # whether it is already updated during training
                self.per_class_lora_A[f'class_{classes[0]}'].data = self.lambda_a_marouf * self.warmup_lora_a.data + (1-self.lambda_a_marouf) * self.per_class_lora_A[f'class_{classes[0]}'].data
            self.per_class = True
            return

        for c in classes:
            if task_memory._class_counter[c] > 0:  # whether it is already updated during training
                self.per_class_lora_A[f'class_{c}'].data = self.lambda_a_marouf * self.warmup_lora_a.data + (1-self.lambda_a_marouf) * self.per_class_lora_A[f'class_{c}'].data
                continue

            self.per_class_lora_A[f'class_{c}'].data.copy_(self.warmup_lora_a.data)
            
        self.per_class = True


    def forward(self, x: torch.Tensor):
        result = F.linear(x, self.weight, bias=self.bias)

        task_memory = TaskMemory()

        num_classes_task = len(task_memory.task_mapping[task_memory.current_task]) if task_memory.current_task else 1
        do_warmup_a = self.training and num_classes_task > 1 and not self.per_class

        current_classes = task_memory.get_classes()
        A = [self.warmup_lora_a if do_warmup_a else self.per_class_lora_A[k] for k in current_classes if k in self.per_class_lora_A]
        B = [self.shared_lora_b] * len(A)
        
        if not A or not B:
            assert not self.training
            return result

        assert all([torch.equal(B[0], b) for b in B]), "B should be the same for all classes"

        # compose lora_A and lora_B
        lora_A = torch.stack(A) # B x r x n
        lora_B = torch.stack(B) # B x m x r
        
        if not self.training:
            updates = []
            for i in range(lora_A.shape[0]):
                b_i = lora_B[i]  # shape: [m, r]
                a_i = lora_A[i]  # shape: [r, n]
                updates.append((b_i @ a_i))  # shape: [m, n]

            if not updates:
                return result

            combined_lora = torch.stack(updates, dim=0).mean(dim=0)
            # (If you really want to average, do .mean(dim=0) instead of .sum(dim=0))

            # F.linear expects weight shape [out_features, in_features].
            # combined_lora shape [m, n], which matches [out_features, in_features].
            result += F.linear(self.lora_dropout(x), self.scaling * combined_lora, bias=None)
            return result
        else:
            transposed = False

            if x.dim() == 2:
                result += torch.bmm(
                    torch.bmm(self.lora_dropout(x).unsqueeze(1), lora_A.transpose(1, 2)),
                    lora_B.transpose(1, 2)
                ).squeeze(1) * self.scaling
            elif x.dim() == 3:
                if x.shape[0] != lora_A.shape[0]:
                    if x.shape[1] == lora_A.shape[0]:
                        x = x.transpose(0, 1)
                        transposed = True
                    else:
                        raise NotImplementedError(f"Unrecognized shape case: x.shape={x.shape}, lora_A.shape={lora_A.shape}")

                ba_result = ((self.lora_dropout(x) @ lora_A.transpose(1, 2)) @ lora_B.transpose(1, 2)) * self.scaling

                if transposed:
                    ba_result = ba_result.transpose(0, 1)
                result += ba_result
            else:
                raise NotImplementedError("Unrecognized dimension!")

        return result