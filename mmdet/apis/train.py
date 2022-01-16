# Copyright (c) OpenMMLab. All rights reserved.
import random
import warnings

import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer,
                         build_runner, get_dist_info)

from mmdet.core import DistEvalHook, EvalHook
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import find_latest_checkpoint, get_root_logger

from mmdet.core.utils.my_misc import d0a, global_vars
from mmdet.core.utils.my_hook_2 import SaveLayerOutput


def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.

    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.

    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None):
    logger = get_root_logger(log_level=cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    if 'imgs_per_gpu' in cfg.data:
        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
                       'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    runner_type = 'EpochBasedRunner' if 'runner' not in cfg else cfg.runner[
        'type']
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # `num_gpus` will be ignored if distributed
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            runner_type=runner_type,
            persistent_workers=cfg.data.get('persistent_workers', False))
        for ds in dataset
    ]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if 'runner' not in cfg:
        cfg.runner = {
            'type': 'EpochBasedRunner',
            'max_epochs': cfg.total_epochs
        }
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)
    else:
        if 'total_epochs' in cfg:
            assert cfg.total_epochs == cfg.runner.max_epochs

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None),
        custom_hooks_config=cfg.get('custom_hooks', None))

    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        # Support batch_size > 1 in validation
        val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
        if val_samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
        # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
        runner.register_hook(
            eval_hook(val_dataloader, **eval_cfg), priority='LOW')

    resume_from = None
    if cfg.resume_from is None and cfg.get('auto_resume'):
        resume_from = find_latest_checkpoint(cfg.work_dir)
    if resume_from is not None:
        cfg.resume_from = resume_from

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        # qqq as in my_hook

        if cfg.mesima_1:
            # https://cv-tricks.com/keras/understand-implement-resnets/
            # https://discuss.pytorch.org/t/accessing-layers-inside-bottleneck-module-of-pretrained-resnet-model/16287
            nsublayer = 0
            for sublayer in runner.model.module.backbone.layer4.children():
                print('--------------- nsublayer=' + str(nsublayer))
                print(sublayer)
                nsublayer += 1
                if nsublayer == 2:
                    sublayer.conv2.weight.shape
                    #   Out[35]: torch.Size([512, 512, 3, 3])
                    nflt = 99
                    ninpchallels = 0, 299, 499
                    for ninpchallel in ninpchallels:
                        flt = sublayer.conv2.weight[ninpchallel, nflt, :,:].cpu().detach().numpy()
                        d0a(flt,save_to_and_close_the_pdf = 1, di_sav_dbg = '/home/konstak/projects2/mmdetection/demo', \
                                pdf_prefix = '', fn = 'before__layer4_' + 'nsublayer=' + str(nsublayer) + '_nflt=' + str(nflt) + '_ninpchallel=' + str(ninpchallel) )
        #if cfg.mesima_1:

        if cfg.mesima_2:
            save_layer_output = SaveLayerOutput()
            hook_handles = []
            for layer in model.modules():
                if 0:
                    if isinstance(layer, torch.nn.modules.conv.Conv2d):
                        handle = layer.register_forward_hook(save_layer_output)
                        hook_handles.append(handle)
                elif 1:
                    if 'rpn_head' in str(type(layer)):
                        handle = layer.register_forward_hook(save_layer_output)
                        hook_handles.append(handle)
                    global_vars.save_layer_output = save_layer_output
        #if cfg.mesima_2:

        runner.load_checkpoint(cfg.load_from)

        if cfg.mesima_1:
            nsublayer = 0
            for sublayer in runner.model.module.backbone.layer4.children():
                print('--------------- nsublayer=' + str(nsublayer))
                print(sublayer)
                nsublayer += 1
                if nsublayer == 2:
                    sublayer.conv2.weight.shape
                    #   Out[35]: torch.Size([512, 512, 3, 3])
                    nflt = 99
                    ninpchallels = 0, 299, 499
                    for ninpchallel in ninpchallels:
                        flt = sublayer.conv2.weight[ninpchallel, nflt, :,:].cpu().detach().numpy()
                        d0a(flt,save_to_and_close_the_pdf = 1, di_sav_dbg = '/home/konstak/projects2/mmdetection/demo', \
                                pdf_prefix = '', fn = 'after__layer4_' + 'nsublayer=' + str(nsublayer) + '_nflt=' + str(nflt) + '_ninpchallel=' + str(ninpchallel) )
        #if cfg.mesima_1:




        tmp=10
    runner.run(data_loaders, cfg.workflow)
