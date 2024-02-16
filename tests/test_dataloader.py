from loguru import logger

import sys
from loguru import logger
import os
from argparse import ArgumentParser

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from tools.run_net import load_config, parse_args
from slowfast.datasets.epickitchens import EpicKitchens


def run(args):
    cfg = load_config(args)

    cfg.TRAIN.BATCH_SIZE = 7

    a = EpicKitchens(cfg=cfg, mode="train")
    logger.debug(f"{len(a[0])=}")
    logger.debug(a[0][0][0].shape)
    logger.debug(a[0][0][1].shape)
    logger.debug(a[0][2])
    logger.debug(a[0][3])


if __name__ == "__main__":
    args = parse_args()

    run(args)
