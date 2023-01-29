#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
import os
import sys
sys.path.append('/media/satya/740516b2-56c1-4cf3-9ae2-87b56289b807/work/project/segmentation/detectron2/projects/TensorMask/tensormask/')

from tensormask import TensorMask

class TensorMaskTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)
