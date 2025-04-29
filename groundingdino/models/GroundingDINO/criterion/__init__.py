# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .criterion import SetCriterion
from .base_criterion import BaseCriterion
from .two_stage_criterion import TwoStageCriterion
from ..matcher import build_matcher
import copy

def build_criterion(args):
    weight_dict={
        "loss_class": 1,
        "loss_bbox": 5.0,
        "loss_giou": 2.0,
    }

    # set aux loss weight dict
    base_weight_dict = copy.deepcopy(weight_dict)
    if args.aux_loss:
        aux_weight_dict = {}
        aux_weight_dict.update({k + "_enc": v for k, v in base_weight_dict.items()})
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in base_weight_dict.items()})
        weight_dict.update(aux_weight_dict)
        
    return TwoStageCriterion(num_classes=args.max_text_len,
                      matcher=build_matcher(args),
                      weight_dict=weight_dict)
