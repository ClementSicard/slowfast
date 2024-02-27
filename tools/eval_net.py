#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Eval a video classification model."""

import numpy as np
from scipy.stats import gmean
import pprint
import torch
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats
from tqdm import tqdm

import slowfast.utils.distributed as du
import slowfast.utils.metrics as metrics


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()

    for cur_iter, (inputs, labels, _, meta) in enumerate(val_loader):
        # Transferthe data to the current GPU device.
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)
        if isinstance(labels, (dict,)):
            labels = {k: v.cuda() for k, v in labels.items()}
        else:
            labels = labels.cuda()

        preds = model(inputs)
        # Compute the verb accuracies.
        verb_top1_acc, verb_top5_acc = metrics.topk_accuracies(preds[0], labels["verb"], (1, 5))

        # Combine the errors across the GPUs.
        if cfg.NUM_GPUS > 1:
            verb_top1_acc, verb_top5_acc = du.all_reduce([verb_top1_acc, verb_top5_acc])

        # Copy the errors from GPU to CPU (sync point).
        verb_top1_acc, verb_top5_acc = verb_top1_acc.item(), verb_top5_acc.item()

        # Compute the noun accuracies.
        noun_top1_acc, noun_top5_acc = metrics.topk_accuracies(preds[1], labels["noun"], (1, 5))

        # Combine the errors across the GPUs.
        if cfg.NUM_GPUS > 1:
            noun_top1_acc, noun_top5_acc = du.all_reduce([noun_top1_acc, noun_top5_acc])

        # Copy the errors from GPU to CPU (sync point).
        noun_top1_acc, noun_top5_acc = noun_top1_acc.item(), noun_top5_acc.item()

        # Compute the action accuracies.
        action_top1_acc, action_top5_acc = metrics.multitask_topk_accuracies(
            (preds[0], preds[1]), (labels["verb"], labels["noun"]), (1, 5)
        )
        # Combine the errors across the GPUs.
        if cfg.NUM_GPUS > 1:
            action_top1_acc, action_top5_acc = du.all_reduce([action_top1_acc, action_top5_acc])

        # Copy the errors from GPU to CPU (sync point).
        action_top1_acc, action_top5_acc = action_top1_acc.item(), action_top5_acc.item()

        val_meter.iter_toc()
        # Update and log stats.
        val_meter.update_stats(
            (verb_top1_acc, noun_top1_acc, action_top1_acc),
            (verb_top5_acc, noun_top5_acc, action_top5_acc),
            inputs[0].size(0) * cfg.NUM_GPUS,
        )

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()
    # Log epoch stats.
    is_best_epoch = val_meter.log_epoch_stats(cur_epoch)
    val_meter.reset()
    return is_best_epoch


@torch.no_grad()
def eval_epoch_gru(val_loader, model, val_meter, cur_epoch, cfg):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()

    for cur_iter, batch in enumerate(
        tqdm(
            val_loader,
            total=len(val_loader),
            desc="Testing",
            unit="batch",
        )
    ):
        inputs, lengths, labels, video_idx, meta = batch

        # Transferthe data to the current GPU device.
        if cfg.NUM_GPUS > 0:
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)

            labels = {k: v.cuda() for k, v in labels.items()}

        preds = model(inputs, lengths=lengths)

        # Compute the verb accuracies.
        verb_top1_acc, verb_top5_acc = metrics.topk_accuracies(preds[0], labels["verb"], (1, 5))

        # Combine the errors across the GPUs.
        if cfg.NUM_GPUS > 1:
            verb_top1_acc, verb_top5_acc = du.all_reduce([verb_top1_acc, verb_top5_acc])

        # Copy the errors from GPU to CPU (sync point).
        verb_top1_acc, verb_top5_acc = verb_top1_acc.item(), verb_top5_acc.item()

        # Compute the noun accuracies.
        noun_top1_acc, noun_top5_acc = metrics.topk_accuracies(preds[1], labels["noun"], (1, 5))

        # Combine the errors across the GPUs.
        if cfg.NUM_GPUS > 1:
            noun_top1_acc, noun_top5_acc = du.all_reduce([noun_top1_acc, noun_top5_acc])

        # Copy the errors from GPU to CPU (sync point).
        noun_top1_acc, noun_top5_acc = noun_top1_acc.item(), noun_top5_acc.item()

        # Compute the action accuracies.
        action_top1_acc, action_top5_acc = metrics.multitask_topk_accuracies(
            (preds[0], preds[1]), (labels["verb"], labels["noun"]), (1, 5)
        )
        # Combine the errors across the GPUs.
        if cfg.NUM_GPUS > 1:
            action_top1_acc, action_top5_acc = du.all_reduce([action_top1_acc, action_top5_acc])

        # Copy the errors from GPU to CPU (sync point).
        action_top1_acc, action_top5_acc = action_top1_acc.item(), action_top5_acc.item()

        val_meter.iter_toc()
        # Update and log stats.
        val_meter.update_stats(
            (verb_top1_acc, noun_top1_acc, action_top1_acc),
            (verb_top5_acc, noun_top5_acc, action_top5_acc),
            inputs[0].size(0) * cfg.NUM_GPUS,
        )

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    is_best_epoch = val_meter.log_epoch_stats(cur_epoch)
    val_meter.reset()

    return is_best_epoch
