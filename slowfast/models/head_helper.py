#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""ResNe(X)t Head helper."""

from loguru import logger
import torch
import torch.nn as nn


class ResNetBasicHead(nn.Module):
    """
    ResNe(X)t 3D head.
    This layer performs a fully-connected projection during training, when the
    input size is 1x1x1. It performs a convolutional projection during testing
    when the input size is larger than 1x1x1. If the inputs are from multiple
    different pathways, the inputs will be concatenated after pooling.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        pool_size,
        dropout_rate=0.0,
        act_func="softmax",
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetBasicHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(ResNetBasicHead, self).__init__()
        assert len({len(pool_size), len(dim_in)}) == 1, "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)

        for pathway in range(self.num_pathways):
            avg_pool = nn.AvgPool3d(pool_size[pathway], stride=1)
            self.add_module("pathway{}_avgpool".format(pathway), avg_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        assert (
            len(num_classes) == 2
        ), f"num_classes must be a list of length 2 but was {len(num_classes)}: {num_classes}"

        self.num_classes = num_classes
        self.dim_in = dim_in
        F = sum(self.dim_in)
        V, N = self.num_classes

        self.projection_verb = nn.Linear(F, V, bias=True)
        self.projection_noun = nn.Linear(F, N, bias=True)
        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError("{} is not supported as an activation function.".format(act_func))

    def forward(self, inputs):
        logger.debug(f"Input: {inputs.shape if isinstance(inputs, torch.Tensor) else [i.shape for i in inputs]}")

        assert len(inputs) == self.num_pathways, "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            pool_out.append(m(inputs[pathway]))

        logger.debug(f"Cat: {pool_out.shape if isinstance(pool_out, torch.Tensor) else [i.shape for i in pool_out]}")
        x = torch.cat(pool_out, 1)

        # (B, C, T, H, W) -> (B, T, H, W, C).
        logger.debug(f"Permute: {x.shape if isinstance(x, torch.Tensor) else [i.shape for i in x]}")
        x = x.permute((0, 2, 3, 4, 1))

        logger.debug(f"After permute: {x.shape if isinstance(x, torch.Tensor) else [i.shape for i in x]}")

        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)

        x_v = self.projection_verb(x)
        x_n = self.projection_noun(x)

        # Performs fully convlutional inference.
        x_v = self.fc_inference(x_v, self.act)
        x_n = self.fc_inference(x_n, self.act)
        return (x_v, x_n)

    def fc_inference(self, x: torch.Tensor, act: nn.Module) -> torch.Tensor:
        """
        Perform fully convolutional inference.

        Args:
            x (tensor): input tensor.
            act (nn.Module): activation function.

        Returns:
            tensor: output tensor.
        """
        if not self.training:
            x = act(x)
            x = x.mean([1, 2, 3])

        x = x.view(x.shape[0], -1)

        return x
