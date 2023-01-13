import importlib

import torch
import numpy as np
from collections import abc
from einops import rearrange
from functools import partial

import multiprocessing as mp
from threading import Thread
from queue import Queue

from inspect import isfunction
from PIL import Image, ImageDraw, ImageFont

import re
import os
import sys
import shutil
from copy import deepcopy
from pathlib import Path
from datetime import datetime
from torch.utils.data import Dataset, ConcatDataset, ChainDataset, IterableDataset
from abc import abstractmethod


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config, **kwargs):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()), **kwargs)


class Txt2ImgIterableBaseDataset(IterableDataset):
    '''
    Define an interface to make the IterableDatasets for text2img data chainable
    '''
    def __init__(self, num_records=0, valid_ids=None, size=256):
        super().__init__()
        self.num_records = num_records
        self.valid_ids = valid_ids
        self.sample_ids = valid_ids
        self.size = size

        print(f'{self.__class__.__name__} dataset contains {self.__len__()} examples.')

    def __len__(self):
        return self.num_records

    @abstractmethod
    def __iter__(self):
        pass


class DummyDataset(Dataset):
    def __init__(self, num_records=100, latent_shape=[3, 20, 28, 20]):
        super().__init__()
        self.num_records = num_records
        self.latent_shape = latent_shape
    
    def __len__(self):
        return self.num_records
    
    def __getitem__(self, idx):
        example = {
            'pixel_values': torch.zeros(*self.latent_shape),
            'conditioning': torch.zeros(4),
        }
        return example
