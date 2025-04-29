from groundingdino.config import get_config
from groundingdino.config.configs.common.coco_schedule import modified_coco_scheduler

# get default config
iter_per_epoch = 20
dataloader = get_config("common/data/odinw_1shot/aerialmaritimedrone_tiled.py").dataloader
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = modified_coco_scheduler(4, 2, base_steps=iter_per_epoch)
train = get_config("common/train.py").train

# modify training config
train.output_dir = "odinw13_1shot/aerialmaritimedrone_tiled_cet"
train.max_iter = 4 * iter_per_epoch
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2
train.seed = 42
train.eval_period = iter_per_epoch * 4
train.checkpointer=dict(period=iter_per_epoch * 4, max_to_keep=100)
train.fast_dev_run.enabled = False

# modify optimizer config
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1
optimizer.lr = 0.001

# modify dataloader config
dataloader.train.dataset.filter_empty = False
dataloader.train.num_workers = 4
dataloader.train.total_batch_size = 2

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir