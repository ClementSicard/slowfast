#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Data loader."""

import itertools
from typing import Any, List
import numpy as np
import pandas as pd
import torch
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

from .build import build_dataset


def detection_collate(batch):
    """
    Collate function for detection task. Concatanate bboxes, labels and
    metadata from different samples in the first dimension instead of
    stacking them to have a batch-size dimension.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated detection data batch.
    """
    inputs, labels, video_idx, extra_data = zip(*batch)
    inputs, video_idx = default_collate(inputs), default_collate(video_idx)
    labels = torch.tensor(np.concatenate(labels, axis=0)).float()

    collated_extra_data = {}
    for key in extra_data[0].keys():
        data = [d[key] for d in extra_data]
        if key == "boxes" or key == "ori_boxes":
            # Append idx info to the bboxes before concatenating them.
            bboxes = [
                np.concatenate([np.full((data[i].shape[0], 1), float(i)), data[i]], axis=1) for i in range(len(data))
            ]
            bboxes = np.concatenate(bboxes, axis=0)
            collated_extra_data[key] = torch.tensor(bboxes).float()
        elif key == "metadata":
            collated_extra_data[key] = torch.tensor(list(itertools.chain(*data))).view(-1, 2)
        else:
            collated_extra_data[key] = default_collate(data)

    return inputs, labels, video_idx, collated_extra_data


def epickitchens_collate(batch: List[Any]):
    frames, labels, indices, metadata = zip(*batch)

    slow_frames, fast_frames = zip(*frames)

    lengths = [slow_frame.shape[0] for slow_frame in slow_frames]
    padded_slow_frames = torch.nn.utils.rnn.pad_sequence(
        slow_frames,
        batch_first=True,
        padding_value=0.0,
    )
    padded_fast_frames = torch.nn.utils.rnn.pad_sequence(
        fast_frames,
        batch_first=True,
        padding_value=0.0,
    )

    padded_frames = (padded_slow_frames, padded_fast_frames)

    indices = torch.tensor(indices)

    grouped_labels = {
        k: torch.stack(v, dim=0) if isinstance(v[0], torch.Tensor) else torch.tensor(v)
        for k, v in pd.DataFrame(labels).to_dict("list").items()
    }

    metadata = pd.DataFrame(metadata).to_dict("list")

    return_tuple = (
        padded_frames,
        lengths,
        grouped_labels,
        indices,
        metadata,
    )

    return return_tuple


def construct_loader(cfg, split):
    """
    Constructs the data loader for the given dataset.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
    """
    assert split in ["train", "val", "test", "train+val"]
    if split in ["train", "train+val"]:
        dataset_name = cfg.TRAIN.DATASET
        batch_size = int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS) if cfg.NUM_GPUS > 0 else cfg.TRAIN.BATCH_SIZE
        shuffle = True
        drop_last = True
    elif split in ["val"]:
        dataset_name = cfg.TRAIN.DATASET
        batch_size = int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS) if cfg.NUM_GPUS > 0 else cfg.TRAIN.BATCH_SIZE
        shuffle = False
        drop_last = False
    elif split in ["test"]:
        dataset_name = cfg.TEST.DATASET
        batch_size = int(cfg.TEST.BATCH_SIZE / cfg.NUM_GPUS) if cfg.NUM_GPUS > 0 else cfg.TEST.BATCH_SIZE
        shuffle = False
        drop_last = False

    # Construct the dataset
    dataset = build_dataset(dataset_name, cfg, split)
    # Create a sampler for multi-process training
    sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None
    # Create a loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=drop_last,
        collate_fn=epickitchens_collate if "GRU" in cfg.MODEL.MODEL_NAME else default_collate,
    )
    return loader


def shuffle_dataset(loader, cur_epoch):
    """ "
    Shuffles the data.
    Args:
        loader (loader): data loader to perform shuffle.
        cur_epoch (int): number of the current epoch.
    """
    assert isinstance(loader.sampler, (RandomSampler, DistributedSampler)), "Sampler type '{}' not supported".format(
        type(loader.sampler)
    )
    # RandomSampler handles shuffling automatically
    if isinstance(loader.sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        loader.sampler.set_epoch(cur_epoch)
