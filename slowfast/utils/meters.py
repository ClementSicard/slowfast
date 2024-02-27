#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Meters."""

import datetime
import numpy as np
from collections import deque
import torch
from fvcore.common.timer import Timer
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
import wandb
from loguru import logger


class ScalarMeter(object):
    """
    A scalar meter uses a deque to track a series of scaler values with a given
    window size. It supports calculating the median and average values of the
    window, and also supports calculating the global average.
    """

    def __init__(self, window_size):
        """
        Args:
            window_size (int): size of the max length of the deque.
        """
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def reset(self):
        """
        Reset the deque.
        """
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def add_value(self, value):
        """
        Add a new scalar value to the deque.
        """
        self.deque.append(value)
        self.count += 1
        self.total += value

    def get_win_median(self):
        """
        Calculate the current median value of the deque.
        """
        return np.median(self.deque)

    def get_win_avg(self):
        """
        Calculate the current average value of the deque.
        """
        return np.mean(self.deque)

    def get_global_avg(self):
        """
        Calculate the global mean value.
        """
        return self.total / self.count


class EPICTrainMeter(object):
    """
    Measure training stats.
    """

    def __init__(self, epoch_iters, cfg):
        """
        Args:
            epoch_iters (int): the overall number of iterations of one epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.epoch_iters = epoch_iters
        self.MAX_EPOCH = cfg.SOLVER.MAX_EPOCH * epoch_iters
        self.iter_timer = Timer()
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_total = 0.0
        self.loss_verb = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_verb_total = 0.0
        self.loss_noun = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_noun_total = 0.0
        self.lr = None
        # Current minibatch accuracies (smoothed over a window).
        self.mb_top1_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_verb_top1_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_verb_top5_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_noun_top1_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_noun_top5_acc = ScalarMeter(cfg.LOG_PERIOD)
        # Number of correctly classified examples.
        self.num_top1_cor = 0
        self.num_top5_cor = 0
        self.num_verb_top1_cor = 0
        self.num_verb_top5_cor = 0
        self.num_noun_top1_cor = 0
        self.num_noun_top5_cor = 0
        self.num_samples = 0

    def reset(self):
        """
        Reset the Meter.
        """
        self.loss.reset()
        self.loss_total = 0.0
        self.loss_verb.reset()
        self.loss_verb_total = 0.0
        self.loss_noun.reset()
        self.loss_noun_total = 0.0
        self.lr = None
        self.mb_top1_acc.reset()
        self.mb_top5_acc.reset()
        self.mb_verb_top1_acc.reset()
        self.mb_verb_top5_acc.reset()
        self.mb_noun_top1_acc.reset()
        self.mb_noun_top5_acc.reset()
        self.num_top1_cor = 0
        self.num_top5_cor = 0
        self.num_verb_top1_cor = 0
        self.num_verb_top5_cor = 0
        self.num_noun_top1_cor = 0
        self.num_noun_top5_cor = 0
        self.num_samples = 0

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()

    def update_stats(self, top1_acc, top5_acc, loss, lr, mb_size):
        """
        Update the current stats.
        Args:
            top1_acc (float): top1 accuracy rate.
            top5_acc (float): top5 accuracy rate.
            loss (float): loss value.
            lr (float): learning rate.
            mb_size (int): mini batch size.
        """
        # Current minibatch stats
        self.mb_verb_top1_acc.add_value(top1_acc[0])
        self.mb_verb_top5_acc.add_value(top5_acc[0])
        self.mb_noun_top1_acc.add_value(top1_acc[1])
        self.mb_noun_top5_acc.add_value(top5_acc[1])
        self.mb_top1_acc.add_value(top1_acc[2])
        self.mb_top5_acc.add_value(top5_acc[2])
        self.loss_verb.add_value(loss[0])
        self.loss_noun.add_value(loss[1])
        self.loss.add_value(loss[2])
        self.lr = lr
        # Aggregate stats
        self.num_verb_top1_cor += top1_acc[0] * mb_size
        self.num_verb_top5_cor += top5_acc[0] * mb_size
        self.num_noun_top1_cor += top1_acc[1] * mb_size
        self.num_noun_top5_cor += top5_acc[1] * mb_size
        self.num_top1_cor += top1_acc[2] * mb_size
        self.num_top5_cor += top5_acc[2] * mb_size
        self.loss_verb_total += loss[0] * mb_size
        self.loss_noun_total += loss[1] * mb_size
        self.loss_total += loss[2] * mb_size
        self.num_samples += mb_size

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (self.MAX_EPOCH - (cur_epoch * self.epoch_iters + cur_iter + 1))
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        mem_usage = misc.gpu_mem_usage()
        stats = {
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "Train/Acc@1 Verb": self.mb_verb_top1_acc.get_win_median(),
            "Train/Acc@5 Verb": self.mb_verb_top5_acc.get_win_median(),
            "Train/Acc@1 Noun": self.mb_noun_top1_acc.get_win_median(),
            "Train/Acc@5 Noun": self.mb_noun_top5_acc.get_win_median(),
            "Train/Acc@1 Action": self.mb_top1_acc.get_win_median(),
            "Train/Acc@5 Action": self.mb_top5_acc.get_win_median(),
            "Train/Loss/Verb Loss": self.loss_verb.get_win_median(),
            "Train/Loss/Noun Loss": self.loss_noun.get_win_median(),
            "Train/Loss/Action Loss": self.loss.get_win_median(),
            "Learning Rate": self.lr,
            "mem": int(np.ceil(mem_usage)),
        }
        logging.log_json_stats(stats)
        wandb.log(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        eta_sec = self.iter_timer.seconds() * (self.MAX_EPOCH - (cur_epoch + 1) * self.epoch_iters)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        mem_usage = misc.gpu_mem_usage()
        verb_top1_acc = self.num_verb_top1_cor / self.num_samples
        verb_top5_acc = self.num_verb_top5_cor / self.num_samples
        noun_top1_acc = self.num_noun_top1_cor / self.num_samples
        noun_top5_acc = self.num_noun_top5_cor / self.num_samples
        top1_acc = self.num_top1_cor / self.num_samples
        top5_acc = self.num_top5_cor / self.num_samples
        avg_loss_verb = self.loss_verb_total / self.num_samples
        avg_loss_noun = self.loss_noun_total / self.num_samples
        avg_loss = self.loss_total / self.num_samples
        stats = {
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "Epoch/Train/Acc@1 Verb": verb_top1_acc,
            "Epoch/Train/Acc@5 Verb": verb_top5_acc,
            "Epoch/Train/Acc@1 Noun": noun_top1_acc,
            "Epoch/Train/Acc@5 Noun": noun_top5_acc,
            "Epoch/Train/Acc@1 Action": top1_acc,
            "Epoch/Train/Acc@5 Action": top5_acc,
            "Loss/Verb Loss": avg_loss_verb,
            "Loss/Noun Loss": avg_loss_noun,
            "Loss/Action Loss": avg_loss,
            "Learning Rate": self.lr,
            "mem": int(np.ceil(mem_usage)),
        }
        logging.log_json_stats(stats)
        wandb.log(stats)


class EPICValMeter(object):
    """
    Measures validation stats.
    """

    def __init__(self, max_iter, cfg):
        """
        Args:
            max_iter (int): the max number of iteration of the current epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.max_iter = max_iter
        self.iter_timer = Timer()
        # Current minibatch accuracies (smoothed over a window).
        self.mb_top1_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_verb_top1_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_verb_top5_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_noun_top1_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_noun_top5_acc = ScalarMeter(cfg.LOG_PERIOD)
        # Max accuracies (over the full val set).
        self.max_top1_acc = 0.0
        self.max_top5_acc = 0.0
        self.max_verb_top1_acc = 0.0
        self.max_verb_top5_acc = 0.0
        self.max_noun_top1_acc = 0.0
        self.max_noun_top5_acc = 0.0
        # Number of correctly classified examples.
        self.num_top1_cor = 0
        self.num_top5_cor = 0
        self.num_verb_top1_cor = 0
        self.num_verb_top5_cor = 0
        self.num_noun_top1_cor = 0
        self.num_noun_top5_cor = 0
        self.num_samples = 0

    def reset(self):
        """
        Reset the Meter.
        """
        self.iter_timer.reset()
        self.mb_top1_acc.reset()
        self.mb_top5_acc.reset()
        self.mb_verb_top1_acc.reset()
        self.mb_verb_top5_acc.reset()
        self.mb_noun_top1_acc.reset()
        self.mb_noun_top5_acc.reset()
        self.num_top1_cor = 0
        self.num_top5_cor = 0
        self.num_verb_top1_cor = 0
        self.num_verb_top5_cor = 0
        self.num_noun_top1_cor = 0
        self.num_noun_top5_cor = 0
        self.num_samples = 0

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()

    def update_stats(self, top1_acc, top5_acc, mb_size):
        """
        Update the current stats.
        Args:
            top1_acc (float): top1 accuracy rate.
            top5_acc (float): top5 accuracy rate.
            mb_size (int): mini batch size.
        """
        self.mb_verb_top1_acc.add_value(top1_acc[0])
        self.mb_verb_top5_acc.add_value(top5_acc[0])
        self.mb_noun_top1_acc.add_value(top1_acc[1])
        self.mb_noun_top5_acc.add_value(top5_acc[1])
        self.mb_top1_acc.add_value(top1_acc[2])
        self.mb_top5_acc.add_value(top5_acc[2])
        self.num_verb_top1_cor += top1_acc[0] * mb_size
        self.num_verb_top5_cor += top5_acc[0] * mb_size
        self.num_noun_top1_cor += top1_acc[1] * mb_size
        self.num_noun_top5_cor += top5_acc[1] * mb_size
        self.num_top1_cor += top1_acc[2] * mb_size
        self.num_top5_cor += top5_acc[2] * mb_size
        self.num_samples += mb_size

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (self.max_iter - cur_iter - 1)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        mem_usage = misc.gpu_mem_usage()
        stats = {
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.max_iter),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "Val/Acc@1 Verb": self.mb_verb_top1_acc.get_win_median(),
            "Val/Acc@5 Verb": self.mb_verb_top5_acc.get_win_median(),
            "Val/Acc@1 Noun": self.mb_noun_top1_acc.get_win_median(),
            "Val/Acc@5 Noun": self.mb_noun_top5_acc.get_win_median(),
            "Val/Acc@1 Action": self.mb_top1_acc.get_win_median(),
            "Val/Acc@5 Action": self.mb_top5_acc.get_win_median(),
            "mem": int(np.ceil(mem_usage)),
        }
        logging.log_json_stats(stats)
        wandb.log(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        verb_top1_acc = self.num_verb_top1_cor / self.num_samples
        verb_top5_acc = self.num_verb_top5_cor / self.num_samples
        noun_top1_acc = self.num_noun_top1_cor / self.num_samples
        noun_top5_acc = self.num_noun_top5_cor / self.num_samples
        top1_acc = self.num_top1_cor / self.num_samples
        top5_acc = self.num_top5_cor / self.num_samples
        self.max_verb_top1_acc = max(self.max_verb_top1_acc, verb_top1_acc)
        self.max_verb_top5_acc = max(self.max_verb_top5_acc, verb_top5_acc)
        self.max_noun_top1_acc = max(self.max_noun_top1_acc, noun_top1_acc)
        self.max_noun_top5_acc = max(self.max_noun_top5_acc, noun_top5_acc)
        is_best_epoch = top1_acc > self.max_top1_acc
        self.max_top1_acc = max(self.max_top1_acc, top1_acc)
        self.max_top5_acc = max(self.max_top5_acc, top5_acc)
        mem_usage = misc.gpu_mem_usage()
        stats = {
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "time_diff": self.iter_timer.seconds(),
            "Epoch/Val/Acc@1 Verb": verb_top1_acc,
            "Epoch/Val/Acc@5 Verb": verb_top5_acc,
            "Epoch/Val/Acc@1 Noun": noun_top1_acc,
            "Epoch/Val/Acc@5 Noun": noun_top5_acc,
            "Epoch/Val/Acc@1 Action": top1_acc,
            "Epoch/Val/Acc@5 Action": top5_acc,
            "Epoch/Val/Max Acc@1 Verb": self.max_verb_top1_acc,
            "Epoch/Val/Max Acc@5 Verb": self.max_verb_top5_acc,
            "Epoch/Val/Max Acc@1 Noun": self.max_noun_top1_acc,
            "Epoch/Val/Max Acc@5 Noun": self.max_noun_top5_acc,
            "Epoch/Val/Max Acc@1 Action": self.max_top1_acc,
            "Epoch/Val/Max Acc@5 Action": self.max_top5_acc,
            "mem": int(np.ceil(mem_usage)),
        }
        logging.log_json_stats(stats)
        wandb.log(stats)

        return is_best_epoch


class EPICTestMeter(object):
    """
    Perform the multi-view ensemble for testing: each video with an unique index
    will be sampled with multiple clips, and the predictions of the clips will
    be aggregated to produce the final prediction for the video.
    The accuracy is calculated with the given ground truth labels.
    """

    def __init__(self, num_videos, num_clips, num_cls, overall_iters):
        """
        Construct tensors to store the predictions and labels. Expect to get
        num_clips predictions from each video, and calculate the metrics on
        num_videos videos.
        Args:
            num_videos (int): number of videos to test.
            num_clips (int): number of clips sampled from each video for
                aggregating the final prediction for the video.
            num_cls (int): number of classes for each prediction.
            overall_iters (int): overall iterations for testing.
        """

        self.iter_timer = Timer()
        self.num_clips = num_clips
        self.overall_iters = overall_iters
        # Initialize tensors.
        self.verb_video_preds = torch.zeros((num_videos, num_cls[0]))
        self.noun_video_preds = torch.zeros((num_videos, num_cls[1]))
        self.verb_video_labels = torch.zeros((num_videos)).long()
        self.noun_video_labels = torch.zeros((num_videos)).long()
        self.metadata = np.zeros(num_videos, dtype=object)
        self.clip_count = torch.zeros((num_videos)).long()

        self.verb_confusion_matrix = torch.zeros((num_cls[0], num_cls[0]))
        self.noun_confusion_matrix = torch.zeros((num_cls[1], num_cls[1]))

        # Reset metric.
        self.reset()

    def reset(self):
        """
        Reset the metric.
        """
        self.clip_count.zero_()
        self.verb_video_preds.zero_()
        self.verb_video_labels.zero_()
        self.noun_video_preds.zero_()
        self.noun_video_labels.zero_()
        self.verb_confusion_matrix.zero_()
        self.noun_confusion_matrix.zero_()
        self.metadata.fill(0)

    def update_stats(self, preds, labels, metadata, clip_ids):
        """
        Collect the predictions from the current batch and perform on-the-flight
        summation as ensemble.
        Args:
            preds (tensor): predictions from the current batch. Dimension is
                N x C where N is the batch size and C is the channel size
                (num_cls).
            labels (tensor): the corresponding labels of the current batch.
                Dimension is N.
            clip_ids (tensor): clip indexes of the current batch, dimension is
                N.
        """
        for ind in range(preds[0].shape[0]):
            vid_id = int(clip_ids[ind]) // self.num_clips
            verb_label = labels[0][ind]
            noun_label = labels[1][ind]
            self.verb_video_labels[vid_id] = labels[0][ind]
            self.verb_video_preds[vid_id] += preds[0][ind]
            self.noun_video_labels[vid_id] = labels[1][ind]
            self.noun_video_preds[vid_id] += preds[1][ind]
            self.metadata[vid_id] = metadata["narration_id"][ind]
            self.clip_count[vid_id] += 1

            # Update confusion matrix
            _, verb_pred = preds[0][ind].max(0)
            _, noun_pred = preds[1][ind].max(0)
            self.verb_confusion_matrix[verb_label, verb_pred] += 1
            self.noun_confusion_matrix[noun_label, noun_pred] += 1

    def log_iter_stats(self, cur_iter):
        """
        Log the stats.
        Args:
            cur_iter (int): the current iteration of testing.
        """
        eta_sec = self.iter_timer.seconds() * (self.overall_iters - cur_iter)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "split": "test_iter",
            "cur_iter": "{}".format(cur_iter + 1),
            "eta": eta,
            "time_diff": self.iter_timer.seconds(),
        }
        logging.log_json_stats(stats)

    def iter_tic(self):
        self.iter_timer.reset()

    def iter_toc(self):
        self.iter_timer.pause()

    def finalize_metrics(self, ks=(1, 5)):
        """
        Calculate and log the final ensembled metrics.
        ks (tuple): list of top-k values for topk_accuracies. For example,
            ks = (1, 5) correspods to top-1 and top-5 accuracy.
        """
        if not all(self.clip_count == self.num_clips):
            logger.warning("clip count {} ~= num clips {}".format(self.clip_count, self.num_clips))
            logger.warning(self.clip_count)

        verb_topks = metrics.topk_accuracies(self.verb_video_preds, self.verb_video_labels, ks)
        noun_topks = metrics.topk_accuracies(self.noun_video_preds, self.noun_video_labels, ks)

        assert len({len(ks), len(verb_topks)}) == 1
        assert len({len(ks), len(noun_topks)}) == 1
        stats = {"split": "test_final"}
        for k, verb_topk in zip(ks, verb_topks):
            stats["verb_top{}_acc".format(k)] = "{:.{prec}f}".format(verb_topk, prec=2)
        for k, noun_topk in zip(ks, noun_topks):
            stats["noun_top{}_acc".format(k)] = "{:.{prec}f}".format(noun_topk, prec=2)
        logging.log_json_stats(stats)
        return (
            (self.verb_video_preds.numpy().copy(), self.noun_video_preds.numpy().copy()),
            (self.verb_video_labels.numpy().copy(), self.noun_video_labels.numpy().copy()),
            (self.verb_confusion_matrix.numpy().copy(), self.noun_confusion_matrix.numpy().copy()),
            self.metadata.copy(),
        )
