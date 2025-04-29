#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import logging
import os
import re
import sys
import time
import torch
import socket
import glob
import json
import random
import datetime
from pathlib import Path
from torch.amp import autocast
from torch.nn.parallel import DataParallel, DistributedDataParallel

from detectron2.config import LazyCall as L
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import SimpleTrainer, default_argument_parser, hooks, launch
from detectron2.engine.defaults import _try_get_key, setup_logger, collect_env_info, _highlight, CfgNode
from detectron2.utils.env import seed_all_rng
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import  print_csv_format, COCOEvaluator
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter, EventStorage

from groundingdino.util import ema
from groundingdino.util.inference import inference_on_dataset
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.lora_utils import get_lora_modules, apply_lora, DetectionLoraCheckpointer
from groundingdino.util.task_memory import TaskMemory
from groundingdino.config.configs.common.data.odinw.mapping_classes import ODINW_13_FILE_MAPPING, ODINW_OVERLAPPED_FILE_MAPPING

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
torch.backends.cudnn.enabled = False

def multi_datasets_setup_logger(args):
    output_dir = args.output_dir
    rank = comm.get_rank()
    if comm.is_main_process() and output_dir:
        PathManager.mkdirs(output_dir)
    setup_logger(output_dir, distributed_rank=rank, name="fvcore")
    logger = setup_logger(output_dir, distributed_rank=rank)
    logger.info(f"Rank of current process: {rank}. World size: {comm.get_world_size()}")
    logger.info("Environment info:\n" + collect_env_info())
    logger.info(f"Command line arguments: {str(args)}")
    return logger

def default_setup(cfg, args):
    output_dir = _try_get_key(cfg, "OUTPUT_DIR", "output_dir", "train.output_dir")
    if comm.is_main_process() and output_dir:
        PathManager.mkdirs(output_dir)

    if comm.is_main_process() and output_dir:
        path = os.path.join(output_dir, "config.yaml")
        if isinstance(cfg, CfgNode):
            with PathManager.open(path, "w") as f:
                f.write(cfg.dump())
        else:
            LazyConfig.save(cfg, path)

    if not (hasattr(args, "eval_only") and args.eval_only):
        torch.backends.cudnn.benchmark = _try_get_key(
            cfg, "CUDNN_BENCHMARK", "train.cudnn_benchmark", default=False
        )

class Trainer(SimpleTrainer):
    def __init__(
        self,
        args,
        model,
        dataloader,
        optimizer,
        amp=False,
        clip_grad_params=None,
        grad_scaler=None,
        batch_size_scale=1,
        categories_names=[],
    ):
        super().__init__(model=model, data_loader=dataloader, optimizer=optimizer)

        unsupported = "AMPTrainer does not support single-process multi-device training!"
        if isinstance(model, DistributedDataParallel):
            assert not (model.device_ids and len(model.device_ids) > 1), unsupported
        assert not isinstance(model, DataParallel), unsupported

        self.args = args

        if amp and grad_scaler is None:
            from torch.cuda.amp import GradScaler
        
            grad_scaler = GradScaler()
        self.grad_scaler = grad_scaler
        self.amp = amp
        self.clip_grad_params = clip_grad_params
        self.batch_size_scale = batch_size_scale
        self.categories_names = categories_names

    def run_step(self):
        assert self.model.training, "[Trainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[Trainer] CUDA is required for AMP training!"

        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        with autocast("cuda", enabled=self.amp):
            loss_dict = self.model(data)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

        if self.amp:
            self.grad_scaler.scale(losses).backward()
            if self.clip_grad_params is not None:
                self.grad_scaler.unscale_(self.optimizer)
                self.clip_grads(self.model.parameters())           
            if self.iter % self.batch_size_scale == 0:
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                self.optimizer.zero_grad()
        else:
            losses.backward()
            if self.clip_grad_params is not None:
                self.clip_grads(self.model.parameters())
            if self.iter % self.batch_size_scale == 0:
                self.optimizer.step()              
                self.optimizer.zero_grad()

        self._write_metrics(loss_dict, data_time)

    def clip_grads(self, params):
        params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return torch.nn.utils.clip_grad_norm_(
                parameters=params,
                **self.clip_grad_params,
            )

    def state_dict(self):
        ret = super().state_dict()
        if self.grad_scaler and self.amp:
            ret["grad_scaler"] = self.grad_scaler.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        if self.grad_scaler and self.amp:
            self.grad_scaler.load_state_dict(state_dict["grad_scaler"])

    def after_train(self):
        if isinstance(self.model, DistributedDataParallel):
            if hasattr(self.model.module, "after_train"):
                self.model.module.after_train()
        elif hasattr(self.model, "after_train"):
            self.model.after_train()

        return super().after_train()
    
    def before_train(self):
        if isinstance(self.model, DistributedDataParallel):
            if hasattr(self.model.module, "before_train"):
                self.model.module.before_train()
        elif hasattr(self.model, "before_train"):
            self.model.before_train()
        return super().before_train()

    def train(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Starting training from iteration {start_iter}")

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter 
        
        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()

                for self.iter in range(start_iter, max_iter):
                    if self.iter == int(max_iter/2):
                        print(">>> END WARMUP, START SPECIALIZATION PHASE <<<")
                        TaskMemory().enable_per_class()
                    self.before_step()
                    self.run_step()
                    self.after_step()
                self.iter += 1
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

class PeriodicCheckpointer(hooks.PeriodicCheckpointer):
    def after_train(self):
        self.step(self.max_iter)
        return super().after_train()

def load_model(model_config_path, model_checkpoint_path, output_dir, eval_mode=False):
    cfg_model = SLConfig.fromfile(model_config_path)
    cfg_model.device = "cuda"
    model = build_model(cfg_model)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)

    if hasattr(cfg_model, 'dithub') and cfg_model.dithub:
        params = []
        lora_params =[]

        model.load_custom_attention()

        lora_modules = get_lora_modules(model, cfg_model.lora_out_min)

        apply_lora(modules=lora_modules, r=cfg_model.lora_r, 
                    lora_alpha=cfg_model.lora_alpha, lora_dropout=cfg_model.lora_dropout)

        lora_params.extend(
            p
            for n, p in model.named_parameters()
            if re.fullmatch('.*lora_.*', n) and p.requires_grad
        )

        params.extend(lora_params)

        for param in model.parameters():
            param.requires_grad = False

        for param in params:
            param.requires_grad = True
        
        if eval_mode:
            if 'model_final.pth' in model_checkpoint_path:
                checkpoint = torch.load(final_model_path)
                model.load_state_dict(checkpoint['model'])
            else:
                last_lora = torch.load(Path(output_dir, 'last_lora.pth'))
                model.load_state_dict(clean_state_dict(last_lora['model']), strict=False)
                final_model_path = Path(os.path.join(output_dir, "model_final.pth"))
                torch.save({'model': model.state_dict()}, final_model_path)

    return model

def do_test(cfg, model, output_dir=None, eval_only=False):
    logger = logging.getLogger("detectron2")

    evaluator = instantiate(L(COCOEvaluator)(
            dataset_name=cfg.dataloader.test.dataset.names,
            output_dir=cfg.train.output_dir,
        )
    )

    if eval_only:
        logger.info("Run evaluation under eval-only mode")
        if cfg.train.model_ema.enabled and cfg.train.model_ema.use_ema_weights_for_eval_only:
            logger.info("Run evaluation with EMA.")
        else:
            logger.info("Run evaluation without EMA.")
        if "evaluator" in cfg.dataloader:
            ret = inference_on_dataset(
                model, instantiate(cfg.dataloader.test), cfg.dataloader.test.dataset.names, evaluator, output_dir=output_dir
            )
            print_csv_format(ret)
        returns = {'bbox': ret['bbox']}
        output_path = Path(cfg.train.output_dir).parent.parent / f'{cfg.dataloader.test.dataset.names}.out'
        with open(output_path, 'a') as f:
            f.write(f'{Path(cfg.train.output_dir).stem} {returns["bbox"]["AP"]}\n')

        return returns
    
    logger.info("Run evaluation without EMA.")
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model, instantiate(cfg.dataloader.test), cfg.dataloader.test.dataset.names, evaluator
        )
        print_csv_format(ret)

        if cfg.train.model_ema.enabled:
            logger.info("Run evaluation with EMA.")
            with ema.apply_model_ema_and_restore(model):
                if "evaluator" in cfg.dataloader:
                    ema_ret = inference_on_dataset(
                        model, instantiate(cfg.dataloader.test), cfg.dataloader.test.dataset.names, evaluator
                    )
                    print_csv_format(ema_ret)
                    ret.update(ema_ret)
        return {'bbox': ret['bbox']}

def do_train(args, cfg):
    config_file = args.model_config_file 
    checkpoint_path = args.model_checkpoint_path 
    model = load_model(config_file, checkpoint_path, output_dir=args.output_dir)
    model.to(cfg.train.device)

    cfg.optimizer.params.model = model
    cfg.optimizer.weight_decay = args.lora_weight_decay
    cfg.optimizer.lr = args.lora_lr
    optim = instantiate(cfg.optimizer)

    train_loader = None
    train_loader = instantiate(cfg.dataloader.train)

    model = create_ddp_model(model, **cfg.train.ddp)
    ema.may_build_model_ema(cfg, model)

    trainer = Trainer(
        args=args,
        model=model,
        dataloader=train_loader,
        optimizer=optim,
        amp=cfg.train.amp.enabled,
        clip_grad_params=cfg.train.clip_grad.params if cfg.train.clip_grad.enabled else None,
        categories_names=cfg.dataloader.train.mapper.categories_names,
    )

    checkpointer = DetectionLoraCheckpointer(
            model,
            cfg.train.output_dir,
            trainer=trainer,
            args=vars(args),
        )

    if comm.is_main_process():
        output_dir = cfg.train.output_dir
        PathManager.mkdirs(output_dir)
        writers = [
            CommonMetricPrinter(cfg.train.max_iter),
            JSONWriter(os.path.join(output_dir, "metrics.json")),
            TensorboardXWriter(output_dir),
        ]

    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            ema.EMAHook(cfg, model) if cfg.train.model_ema.enabled else None,
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer, file_prefix='lora')
            if comm.is_main_process()
            else None,
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
            hooks.PeriodicWriter(
                writers,
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    start_iter = 0

    TaskMemory().current_task = cfg.dataloader.train.dataset.names.split('_')[0].lower()

    trainer.train(start_iter, cfg.train.max_iter)

    TaskMemory().end_task()
    

def main(args):
    seed_all_rng(args.seed)
    logger = multi_datasets_setup_logger(args)
    if args.zero_shot:
        pre_trained_model_path = args.model_checkpoint_path

    config_dirs = args.config_file
    TaskMemory().task_mapping = ODINW_OVERLAPPED_FILE_MAPPING if 'overlapped' in config_dirs else ODINW_13_FILE_MAPPING

    ow_config_files = glob.glob(os.path.join(config_dirs, "for_train", "*.py"))
    if args.shuffle_tasks:
        random.shuffle(ow_config_files)

    start_time = datetime.datetime.now()

    # train
    for _, ow_config_file in enumerate(ow_config_files):
        torch.cuda.empty_cache()
        args.config_file = ow_config_file
        cfg = LazyConfig.load(ow_config_file)
        cfg = LazyConfig.apply_overrides(cfg, args.opts)
        cfg.train.output_dir = os.path.join(args.output_dir, cfg.train.output_dir)
        default_setup(cfg, args)

        if not args.eval_only:
            do_train(args, cfg)

    coco_config_file = os.path.join(config_dirs, "test_zero_shot_coco.py")

    # eval
    args.eval_only = True
    if args.zero_shot:
        args.model_checkpoint_path = pre_trained_model_path

    if os.path.exists(coco_config_file):
        ow_config_files += [coco_config_file]
    json_paths = {}

    for ow_config_file in ow_config_files:
        torch.cuda.empty_cache()
        args.config_file = ow_config_file
        cfg = LazyConfig.load(args.config_file)
        cfg = LazyConfig.apply_overrides(cfg, args.opts)
        cfg.train.output_dir = os.path.join(args.output_dir, cfg.train.output_dir)
        default_setup(cfg, args)

        config_file = args.model_config_file  
        checkpoint_path = args.model_checkpoint_path

        model = load_model(config_file, checkpoint_path, output_dir=args.output_dir, eval_mode=True)
        model.unfreeze_module(model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        ema.may_build_model_ema(cfg, model)
        if cfg.train.model_ema.enabled and cfg.train.model_ema.use_ema_weights_for_eval_only:
            ema.apply_model_ema(model)
        json_path = os.path.join(os.path.join(args.output_dir, cfg.train.output_dir), "result.json")
        json_paths[ow_config_file] = json_path
        res = do_test(cfg, model, args.output_dir, eval_only=True)
        with open(json_path, "w") as jf:
            json.dump(res, jf)

    avg_res = {}
    for k, v in json_paths.items():
        with open(v, "r") as jf:
            res = json.load(jf)
            if 'bbox' in res:
                avg_res[k] = res['bbox']['AP']

    result_str = f"AP results: {avg_res}"
    logger.info(result_str)
    sum_ = 0
    coco_count = 0
    for k, v in avg_res.items():
        if k != coco_config_file:
            sum_ += v
        else:
            coco_count += 1
    logger.info(f"average AP: {sum_ / (len(avg_res) - coco_count)}")
    if coco_config_file in avg_res:
        logger.info(f"AP on COCO: {avg_res[coco_config_file]}")

    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    logger.info(f"Elapsed time: {elapsed_time}")

if __name__ == "__main__":
    print(socket.gethostname())
    parser = default_argument_parser()
    parser.add_argument("--model-config-file", "-c", type=str, required=True, help="path to model config file")
    parser.add_argument("--model-checkpoint-path", "-p", type=str, required=True, help="path to model checkpoint file")
    parser.add_argument("--seed", type=int, default=None, required=False, help="path to model checkpoint file")
    parser.add_argument("--output-dir", type=str, default="output/odinw", required=False, help="path to model checkpoint file")
    parser.add_argument("--shuffle-tasks", action="store_true", help="perform shuffle tasks only")
    parser.add_argument("--zero-shot", action="store_true", help="perform shuffle tasks only")

    parser.add_argument("--dithub", action="store_true", default=False)

    parser.add_argument('--lora-r', type=int, default=8)
    parser.add_argument('--lora-alpha', type=float, default=8)
    parser.add_argument('--lora-dropout', type=float, default=0.0)
    parser.add_argument('--lora-out-min', type=int, default=128)
    parser.add_argument('--lora-lr', type=float, default=1e-4)
    parser.add_argument('--lora-weight-decay', type=float, default=0.0)

    args = parser.parse_args()

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
