# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


def build_model(cfg_model):
    # we use register to maintain models from catdet6 on.
    from .registry import MODULE_BUILD_FUNCS

    assert cfg_model.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(cfg_model.modelname)
    model = build_func(cfg_model)
    return model
