# --------------------------------------------------------
# Modified by Mzero
# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import pdb
import math
import time
import json
import random
import argparse
import datetime
import tqdm
import math
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from utils.lr_scheduler import build_scheduler
from utils.optimizer import build_optimizer
from utils.logger import create_logger
from utils.utils import  NativeScalerWithGradNormCount, auto_resume_helper, reduce_tensor
from utils.utils import load_checkpoint_ema, load_pretrained_ema, save_checkpoint_ema
from utils.utilsKD import HintConv

from RepDistiller.distiller_zoo import DistillKL, Attention, HintLoss
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count

from timm.utils import ModelEma as ModelEma

if torch.multiprocessing.get_start_method() != "spawn":
    print(f"||{torch.multiprocessing.get_start_method()}||", end="")
    torch.multiprocessing.set_start_method("spawn", force=True)


def import_abspy(name="models", path="classification/"):
    import sys
    import importlib
    path = os.path.abspath(path)
    assert os.path.isdir(path)
    sys.path.insert(0, path)
    module = importlib.import_module(name)
    sys.path.pop(0)
    return module


def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, default="", help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', default=time.strftime("%Y%m%d%H%M%S", time.localtime()), help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
    parser.add_argument('--optim', type=str, help='overwrite optimizer if provided, can be adamw/sgd.')

    # EMA related parameters
    parser.add_argument('--model_ema', type=str2bool, default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', type=str2bool, default=False, help='')

    parser.add_argument('--memory_limit_rate', type=float, default=-1, help='limitation of gpu memory use')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config, args):    
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config) 
    
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model = model.cuda()
    
    # initialize teacher model
    model_t = None
    if config.KD.ENABLE:
        from utils.utils import freeze_model
        model_t = build_model(config, is_pretrain=True, get_teacher=True)
        model_t.cuda()
        load_pretrained_ema(config, model_t, logger, load_teacher=True)
        freeze_model(model_t)
        acc1, acc5, loss = validate(config, data_loader_val, model_t)
        logger.info(f"Accuracy of the teacher network on the {len(dataset_val)} test images: {acc1:.1f}%")
        
    if dist.get_rank() == 0:
        if hasattr(model, 'flops'):
            logger.info(str(model))
            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"number of params: {n_parameters}")
            flops = model.flops()
            logger.info(f"number of GFLOPs: {flops / 1e9}")
        else:
            logger.info(flop_count_str(FlopCountAnalysis(model, (dataset_val[0][0][None],))))
                
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)
        
    criterion_dict = torch.nn.ModuleDict({})
    if config.LOSS.CLASSIFICATION.WEIGHT:
        if config.AUG.MIXUP > 0.:
            # smoothing is handled with mixup label transform
            criterion_dict['CLASSIFICATION'] = SoftTargetCrossEntropy()
        elif config.MODEL.LABEL_SMOOTHING > 0.:
            criterion_dict['CLASSIFICATION'] = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
        else:
            criterion_dict['CLASSIFICATION'] = torch.nn.CrossEntropyLoss()
    if config.LOSS.RESPONSE_KL.WEIGHT:
        criterion_dict['RESPONSE_KL'] = DistillKL(config.LOSS.RESPONSE_KL.T)
    if config.LOSS.B_Hint.WEIGHT_STAGE[-1] or config.LOSS.C_Hint.WEIGHT_STAGE[-1] or config.LOSS.dt_Hint.WEIGHT_STAGE[-1] or \
       config.LOSS.h_Hint.WEIGHT_STAGE[-1]:
        criterion_dict['Hint'] = HintLoss()
    if config.LOSS.FEATURE_AT.WEIGHT or \
       config.LOSS.B_AT.WEIGHT_STAGE[-1] or config.LOSS.C_AT.WEIGHT_STAGE[-1] or config.LOSS.dt_AT.WEIGHT_STAGE[-1]:
        criterion_dict['AT'] = Attention()
    
    # trainable moduls
    KD_trainable_dict = torch.nn.ModuleDict({})
    need_KD_trainable_modules = config.LOSS.dt_Hint.WEIGHT_STAGE[-1] or \
                            config.LOSS.h_Hint.WEIGHT_STAGE[-1] or \
                            (config.LOSS.dt_AT.WEIGHT_STAGE[-1] and config.LOSS.dt_AT.REGRESS) or \
                            (config.LOSS.B_Hint.WEIGHT_STAGE[-1] and config.LOSS.B_Hint.REGRESS[0]) or \
                            (config.LOSS.C_Hint.WEIGHT_STAGE[-1] and config.LOSS.C_Hint.REGRESS[0])
    if need_KD_trainable_modules:
        data = torch.randn(2, 3, 224, 224).cuda()
        model_t.eval()
        model.eval()
        _, _, feats_dict_all = model(data, return_all=True)
        _, _, feats_dict_all_t = model_t(data, return_all=True)
        if config.LOSS.dt_Hint.WEIGHT_STAGE[-1] or (config.LOSS.dt_AT.WEIGHT_STAGE[-1] and config.LOSS.dt_AT.REGRESS):
            for layer_idx in feats_dict_all['dts'].keys():  # dts
                D = feats_dict_all['dts'][layer_idx][0].shape[1] / 4
                D_t = feats_dict_all_t['dts'][layer_idx][0].shape[1] / 4
                if config.LOSS.dt_Hint.WEIGHT_STAGE[-1]:
                    assert config.LOSS.dt_Hint.REGRESS[0]
                    hint_regress = HintConv(int(D), int(D_t), separate=config.LOSS.dt_Hint.REGRESS[1])
                    key_name = f"dt_Hint_regress_layer{layer_idx}"
                    KD_trainable_dict[key_name] = hint_regress
                if config.LOSS.dt_AT.WEIGHT_STAGE[-1] and config.LOSS.dt_AT.REGRESS:
                    AT_regress = torch.nn.Conv2d(int(D), int(D_t), kernel_size=1)
                    key_name = f"dt_AT_regress_layer{layer_idx}"
                    KD_trainable_dict[key_name] = AT_regress
        if config.LOSS.h_Hint.WEIGHT_STAGE[-1]:
            for layer_idx in feats_dict_all['h'].keys():  # h
                D = feats_dict_all['h'][layer_idx][0].shape[1] / 4
                D_t = feats_dict_all_t['h'][layer_idx][0].shape[1] / 4
                assert config.LOSS.dt_Hint.REGRESS[0]
                hint_regress = HintConv(int(D), int(D_t), separate=config.LOSS.dt_Hint.REGRESS[1])
                key_name = f"h_Hint_regress_layer{layer_idx}"
                KD_trainable_dict[key_name] = hint_regress
        if config.LOSS.B_Hint.WEIGHT_STAGE[-1] and config.LOSS.B_Hint.REGRESS[0]:
            for layer_idx in feats_dict_all['Bs'].keys():  # Bs
                hint_regress = HintConv(1, 1, separate=config.LOSS.B_Hint.REGRESS[1])
                key_name = f"B_Hint_regress_layer{layer_idx}"
                KD_trainable_dict[key_name] = hint_regress
        if config.LOSS.C_Hint.WEIGHT_STAGE[-1] and config.LOSS.C_Hint.REGRESS[0]:
            for layer_idx in feats_dict_all['Cs'].keys():  # Cs
                hint_regress = HintConv(1, 1, separate=config.LOSS.C_Hint.REGRESS[1])
                key_name = f"C_Hint_regress_layer{layer_idx}"
                KD_trainable_dict[key_name] = hint_regress            
    
    criterion_dict.cuda()
    KD_trainable_dict.cuda()
    
    #  optimizer
    All_trainable_dict = torch.nn.ModuleDict({})
    All_trainable_dict['model'] = model
    All_trainable_dict.update(KD_trainable_dict)
    optimizer = build_optimizer(config, All_trainable_dict, logger)
    
    # DDP initialization
    model_without_ddp = model
    KD_trainable_dict_without_ddp = KD_trainable_dict
    model = torch.nn.parallel.DistributedDataParallel(model, broadcast_buffers=False)
    for value in KD_trainable_dict.values():
        value = torch.nn.parallel.DistributedDataParallel(value, broadcast_buffers=False)
        
    loss_scaler = NativeScalerWithGradNormCount()
    
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
    
    max_accuracy = 0.0
    max_accuracy_ema = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy, max_accuracy_ema = load_checkpoint_ema(config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger, model_ema, 
                                                             KD_trainable_dict=KD_trainable_dict_without_ddp)
        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if model_ema is not None:
            acc1_ema, acc5_ema, loss_ema = validate(config, data_loader_val, model_ema.ema)
            logger.info(f"Accuracy of the network ema on the {len(dataset_val)} test images: {acc1_ema:.1f}%")

        if config.EVAL_MODE:
            return

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained_ema(config, model_without_ddp, logger, model_ema)   # TODO 是否loadKD_trainable_dict_without_ddp
        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if model_ema is not None:
            acc1_ema, acc5_ema, loss_ema = validate(config, data_loader_val, model_ema.ema)
            logger.info(f"Accuracy of the network ema on the {len(dataset_val)} test images: {acc1_ema:.1f}%")
        
        if config.EVAL_MODE:
            return

    if config.THROUGHPUT_MODE and (dist.get_rank() == 0):
        logger.info(f"throughput mode ==============================")
        throughput(data_loader_val, model, logger)
        if model_ema is not None:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            throughput(data_loader_val, model_ema.ema, logger)
        return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)
            
        train_one_epoch(config, model, criterion_dict, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler, model_ema, 
                        model_t=model_t, KD_trainable_dict=KD_trainable_dict)

        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        
        if model_ema is not None:
            acc1_ema, acc5_ema, loss_ema = validate(config, data_loader_val, model_ema.ema)
            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1_ema:.1f}%")
            if dist.get_rank() == 0 and max_accuracy_ema < acc1_ema:
                max_accuracy_ema = acc1_ema
                save_checkpoint_ema(config, epoch, model_without_ddp, max(max_accuracy,acc1), optimizer, lr_scheduler, loss_scaler, logger, model_ema, max_accuracy_ema, 
                                    file_name='best_ema.pth', KD_trainable_dict=KD_trainable_dict_without_ddp)
            logger.info(f'Max accuracy ema: {max_accuracy_ema:.2f}%')
            
        if dist.get_rank() == 0 and max_accuracy < acc1:
            max_accuracy = acc1
            save_checkpoint_ema(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, loss_scaler, logger, model_ema, max_accuracy_ema, 
                                file_name='best.pth', KD_trainable_dict=KD_trainable_dict_without_ddp)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')
        
        if dist.get_rank() == 0:
            save_checkpoint_ema(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, loss_scaler, logger, model_ema, max_accuracy_ema, 
                                file_name='last.pth', KD_trainable_dict=KD_trainable_dict_without_ddp)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, criterion_dict, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler, model_ema=None, model_time_warmup=50, 
                    model_t=None, KD_trainable_dict={}):
    model.train()
    if config.KD.ENABLE:
        for value in KD_trainable_dict.values():
            value.train()
    
    optimizer.zero_grad()
    
    num_steps = len(data_loader)
    batch_time = AverageMeter()
    model_time = AverageMeter()
    data_time = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()
    loss_meter = AverageMeter()
    loss_meters = {}
    for loss_name in config.LOSS:
        if hasattr(config.LOSS[loss_name], 'WEIGHT_STAGE'):
            if config.LOSS[loss_name].WEIGHT_STAGE[-1] != 0:
                loss_meters[loss_name] = AverageMeter()
        elif hasattr(config.LOSS[loss_name], 'WEIGHT'):
            if config.LOSS[loss_name].WEIGHT:
                loss_meters[loss_name] = AverageMeter()

    start = time.time()
    end = time.time()

    for idx, (samples, targets) in enumerate(data_loader):
        # pdb.set_trace()
        loss_dict = {}
        
        torch.cuda.reset_peak_memory_stats()
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        
        data_time.update(time.time() - end)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            if config.KD.ENABLE:
                outputs, x_list, feats_dict_all = model(samples, return_all=True)
                outputs = outputs.to(torch.float32)
                with torch.no_grad() and torch.inference_mode(mode=True):
                    model_t.eval()
                    outputs_t, x_list_t, feats_dict_all_t = model_t(samples, return_all=True)
                    outputs_t = outputs_t.to(torch.float32).detach()
                    x_list_t = [x.detach() for x in x_list_t]
            else:
                outputs = model(samples).to(torch.float32)
        
        for loss_name in loss_meters.keys():
            if loss_name == 'CLASSIFICATION':
                loss_dict['CLASSIFICATION'] = criterion_dict['CLASSIFICATION'](outputs, targets) * config.LOSS.CLASSIFICATION.WEIGHT
            if loss_name == 'RESPONSE_KL':   
                loss_dict['RESPONSE_KL'] = criterion_dict['RESPONSE_KL'](outputs, outputs_t) * config.LOSS.RESPONSE_KL.WEIGHT
            if loss_name == 'FEATURE_AT':
                loss_group = criterion_dict['AT'](x_list, x_list_t)
                loss_dict['FEATURE_AT'] = sum(loss_group) * config.LOSS.FEATURE_AT.WEIGHT
            if loss_name == 'dt_AT':
                K = 4
                loss_dict['dt_AT'] = 0
                feats_dt, feats_dt_t = feats_dict_all['dts'], feats_dict_all_t['dts']
                for layer_idx in feats_dt.keys():   # 每个layer
                    bsz, KD, L = feats_dt[layer_idx][0].shape
                    H = int(math.sqrt(L))
                    for dt, dt_t in zip(feats_dt[layer_idx], feats_dt_t[layer_idx]):      # 每个block
                        if config.LOSS.dt_AT.REGRESS:
                            key_name = f"dt_AT_regress_layer{layer_idx}"
                            with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
                                dt = KD_trainable_dict[key_name](dt.view(bsz*K, -1, H, H))     # [B, K*D, L]  ->   [BK, D, H, W]  -> [BK, D_t, H, W]
                        loss_dict['dt_AT'] += sum(criterion_dict['AT']([dt.view(bsz*K,-1,H,H)], [dt_t.view(bsz*K,-1,H,H)])) * config.LOSS.dt_AT.WEIGHT_STAGE[layer_idx]
            if loss_name == 'dt_Hint':
                K = 4
                loss_dict['dt_Hint'] = 0
                feats_dt, feats_dt_t = feats_dict_all['dts'], feats_dict_all_t['dts']
                for layer_idx in feats_dt.keys():   # 每个layer
                    for dt, dt_t in zip(feats_dt[layer_idx], feats_dt_t[layer_idx]):      # 每个block
                        key_name = f"dt_Hint_regress_layer{layer_idx}"
                        dt_, dt_t_ = dt.clone(), dt_t.detach().clone()
                        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
                            dt_ = KD_trainable_dict[key_name](dt_)  # [B, K*D, L]  ->  [B, K*D_t, L]     
                        loss_dict['dt_Hint'] += criterion_dict['Hint'](dt_, dt_t_) * config.LOSS.dt_Hint.WEIGHT_STAGE[layer_idx]
            if loss_name == 'h_Hint':
                loss_dict['h_Hint'] = 0
                feats_h, feats_h_t = feats_dict_all['h'], feats_dict_all_t['h']
                for layer_idx in feats_h.keys():   # 每个layer
                    for h, h_t in zip(feats_h[layer_idx], feats_h_t[layer_idx]):      # 每个block
                        h_, h_t_ = h.clone(), h_t.detach().clone()
                        if not config.LOSS.h_Hint.h_RANGE[0]:   # 截取特定位置的hidden states
                            indices = [h_.shape[-1] // 4, h_.shape[-1] // 2, 3 * h_.shape[-1] // 4, h_.shape[-1] - 1]
                            selected_h, selected_h_t = [], []
                            for i, control in enumerate(config.LOSS.h_Hint.h_RANGE[1:]):
                                if control:
                                    index = indices[i]
                                    selected_h.append(h_[..., index:index+1])
                                    selected_h_t.append(h_t_[..., index:index+1])
                            h_ = torch.cat(selected_h, dim=-1)
                            h_t_ = torch.cat(selected_h_t, dim=-1)
                        key_name = f"h_Hint_regress_layer{layer_idx}"
                        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
                            h_ = KD_trainable_dict[key_name](h_) 
                        loss_dict['h_Hint'] += criterion_dict['Hint'](h_, h_t_.squeeze(2)) * config.LOSS.h_Hint.WEIGHT_STAGE[layer_idx]
            if loss_name == 'B_AT':
                loss_dict['B_AT'] = 0
                feats_B, feats_B_t = feats_dict_all['Bs'], feats_dict_all_t['Bs']
                for layer_idx in feats_B.keys():   # 每个layer
                    bsz = feats_B[layer_idx][0].shape[0]
                    H = int(math.sqrt(feats_B[layer_idx][0].shape[-1]))
                    for B, B_t in zip(feats_B[layer_idx], feats_B_t[layer_idx]):      # 每个block
                        loss_dict['B_AT'] += sum(criterion_dict['AT']([B.view(bsz*4,-1, H, H)], [B_t.view(bsz*4,-1, H, H)])) * config.LOSS.B_AT.WEIGHT_STAGE[layer_idx]  # [B, K, N, L]   ->   [BK, 1, H, W]
            if loss_name == 'B_Hint':
                loss_dict['B_Hint'] = 0
                feats_B, feats_B_t = feats_dict_all['Bs'], feats_dict_all_t['Bs']
                for layer_idx in feats_B.keys():   # 每个layer
                    for B, B_t in zip(feats_B[layer_idx], feats_B_t[layer_idx]):      # 每个block
                        B_, B_t_ = B.clone(), B_t.detach().clone()
                        if config.LOSS.B_Hint.REGRESS[0]:
                            key_name = f"B_Hint_regress_layer{layer_idx}"
                            with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
                                B_ = KD_trainable_dict[key_name](B_) 
                        loss_dict['B_Hint'] += criterion_dict['Hint'](B_, B_t_.squeeze(2)) * config.LOSS.B_Hint.WEIGHT_STAGE[layer_idx]
            if loss_name == 'C_AT':
                loss_dict['C_AT'] = 0
                feats_C, feats_C_t = feats_dict_all['Cs'], feats_dict_all_t['Cs']
                for layer_idx in feats_C.keys():   # 每个layer
                    bsz = feats_C[layer_idx][0].shape[0]
                    H = int(math.sqrt(feats_C[layer_idx][0].shape[-1]))
                    for C, C_t in zip(feats_C[layer_idx], feats_C_t[layer_idx]):      # 每个block
                        loss_dict['C_AT'] += sum(criterion_dict['AT']([C.view(bsz*4,-1, H, H)], [C_t.view(bsz*4,-1, H, H)])) * config.LOSS.C_AT.WEIGHT_STAGE[layer_idx]     # [B, K, N, L]   ->   [BK, 1, H, W]
            if loss_name == 'C_Hint':
                loss_dict['C_Hint'] = 0
                feats_C, feats_C_t = feats_dict_all['Cs'], feats_dict_all_t['Cs']
                for layer_idx in feats_C.keys():   # 每个layer
                    for C, C_t in zip(feats_C[layer_idx], feats_C_t[layer_idx]):      # 每个block
                        C_, C_t_ = C.clone(), C_t.detach().clone()
                        if config.LOSS.C_Hint.REGRESS[0]:
                            key_name = f"C_Hint_regress_layer{layer_idx}"
                            with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
                                C_ = KD_trainable_dict[key_name](C_)
                        loss_dict['C_Hint'] += criterion_dict['Hint'](C_, C_t_.squeeze(2)) * config.LOSS.C_Hint.WEIGHT_STAGE[layer_idx]
            
            loss_meters[loss_name].update(loss_dict[loss_name], targets.size(0))
        loss = sum(loss_dict.values()) / config.TRAIN.ACCUMULATION_STEPS
        
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
            if model_ema is not None:
                model_ema.update(model)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        if grad_norm is not None:  # loss_scaler return None if not update
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx > model_time_warmup:
            model_time.update(batch_time.val - data_time.val)

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            loss_str = ''
            for loss_name, meter in loss_meters.items():
                loss_str += f'{loss_name} {meter.val:.4f} ({meter.avg:.4f})\t'
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'data time {data_time.val:.4f} ({data_time.avg:.4f})\t'
                f'model time {model_time.val:.4f} ({model_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'{loss_str}\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = model(images)

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    args, config = parse_option()   # 解析命令行参数和配置文件参数
    
    if config.AMP_OPT_LEVEL:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")

    # 初始化分布式训练环境
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    dist.barrier()

    # 设置随机数种子
    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    # to make sure all the config.OUTPUT are the same
    config.defrost()
    if dist.get_rank() == 0:
        obj = [config.OUTPUT]
        # obj = [str(random.randint(0, 100))] # for test
    else:
        obj = [None]
    dist.broadcast_object_list(obj)
    dist.barrier()
    config.OUTPUT = obj[0]
    print(config.OUTPUT, flush=True)
    config.freeze()
    os.makedirs(config.OUTPUT, exist_ok=True)
    if config.KD.ENABLE:
        logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=config.OUTPUT_NAME)
    else:
        logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=config.MODEL.NAME)
    
    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    if args.memory_limit_rate > 0 and args.memory_limit_rate < 1:
        torch.cuda.set_per_process_memory_fraction(args.memory_limit_rate)
        usable_memory = torch.cuda.get_device_properties(0).total_memory * args.memory_limit_rate / 1e6
        print(f"===========> GPU memory is limited to {usable_memory}MB", flush=True)

    main(config, args)
