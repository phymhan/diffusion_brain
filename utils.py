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

from test_tube import Experiment
from argparse import Namespace
from typing import Any, Dict, Optional, Union
import pytorch_lightning as pl
from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_deprecation, rank_zero_warn
from pytorch_lightning.utilities.distributed import rank_zero_only


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


class TestTubeLogger(LightningLoggerBase):
    r"""
    Log to local file system in `TensorBoard <https://www.tensorflow.org/tensorboard>`_ format
    but using a nicer folder structure (see `full docs <https://williamfalcon.github.io/test-tube>`_).

    Warning:
        The test-tube package is no longer maintained and PyTorch Lightning will remove the :class:´TestTubeLogger´
        in v1.7.0.

    Install it with pip:

    .. code-block:: bash

        pip install test_tube

    .. code-block:: python

        from pytorch_lightning import Trainer
        from pytorch_lightning.loggers import TestTubeLogger

        logger = TestTubeLogger("tt_logs", name="my_exp_name")
        trainer = Trainer(logger=logger)

    Use the logger anywhere in your :class:`~pytorch_lightning.core.lightning.LightningModule` as follows:

    .. code-block:: python

        from pytorch_lightning import LightningModule


        class LitModel(LightningModule):
            def training_step(self, batch, batch_idx):
                # example
                self.logger.experiment.whatever_method_summary_writer_supports(...)

            def any_lightning_module_function_or_hook(self):
                self.logger.experiment.add_histogram(...)

    Args:
        save_dir: Save directory
        name: Experiment name. Defaults to ``'default'``.
        description: A short snippet about this experiment
        debug: If ``True``, it doesn't log anything.
        version: Experiment version. If version is not specified the logger inspects the save
            directory for existing versions, then automatically assigns the next available version.
        create_git_tag: If ``True`` creates a git tag to save the code used in this experiment.
        log_graph: Adds the computational graph to tensorboard. This requires that
            the user has defined the `self.example_input_array` attribute in their
            model.
        prefix: A string to put at the beginning of metric keys.

    Raises:
        ModuleNotFoundError:
            If required TestTube package is not installed on the device.
    """

    __test__ = False
    LOGGER_JOIN_CHAR = "-"

    def __init__(
        self,
        save_dir: str,
        name: str = "default",
        description: Optional[str] = None,
        debug: bool = False,
        version: Optional[int] = None,
        create_git_tag: bool = False,
        log_graph: bool = False,
        prefix: str = "",
    ):
        rank_zero_deprecation(
            "The TestTubeLogger is deprecated since v1.5 and will be removed in v1.7. We recommend switching to the"
            " `pytorch_lightning.loggers.TensorBoardLogger` as an alternative."
        )
        if Experiment is None:
            raise ModuleNotFoundError(
                "You want to use `test_tube` logger which is not installed yet,"
                " install it with `pip install test-tube`."
            )
        super().__init__()
        self._save_dir = save_dir
        self._name = name
        self.description = description
        self.debug = debug
        self._version = version
        self.create_git_tag = create_git_tag
        self._log_graph = log_graph
        self._prefix = prefix
        self._experiment = None

    @property
    @rank_zero_experiment
    def experiment(self) -> Experiment:
        r"""

        Actual TestTube object. To use TestTube features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.

        Example::

            self.logger.experiment.some_test_tube_function()

        """
        if self._experiment is not None:
            return self._experiment

        self._experiment = Experiment(
            save_dir=self.save_dir,
            name=self._name,
            debug=self.debug,
            version=self.version,
            description=self.description,
            create_git_tag=self.create_git_tag,
            rank=rank_zero_only.rank,
        )
        return self._experiment

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        # TODO: HACK figure out where this is being set to true
        self.experiment.debug = self.debug
        params = self._convert_params(params)
        params = self._flatten_dict(params)
        self.experiment.argparse(Namespace(**params))

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        # TODO: HACK figure out where this is being set to true
        metrics = self._add_prefix(metrics)
        self.experiment.debug = self.debug
        self.experiment.log(metrics, global_step=step)

    @rank_zero_only
    def log_graph(self, model: "pl.LightningModule", input_array=None):
        if self._log_graph:
            if input_array is None:
                input_array = model.example_input_array

            if input_array is not None:
                self.experiment.add_graph(model, model._apply_batch_transfer_handler(input_array))
            else:
                rank_zero_warn(
                    "Could not log computational graph since neither the"
                    " `model.example_input_array` attribute is set nor"
                    " `input_array` was given",
                    UserWarning,
                )

    @rank_zero_only
    def save(self) -> None:
        super().save()
        # TODO: HACK figure out where this is being set to true
        self.experiment.debug = self.debug
        self.experiment.save()

    @rank_zero_only
    def finalize(self, status: str) -> None:
        super().finalize(status)
        # TODO: HACK figure out where this is being set to true
        self.experiment.debug = self.debug
        self.save()
        self.close()

    @rank_zero_only
    def close(self) -> None:
        super().save()
        # TODO: HACK figure out where this is being set to true
        self.experiment.debug = self.debug
        if not self.debug:
            exp = self.experiment
            exp.close()

    @property
    def save_dir(self) -> Optional[str]:
        """Gets the save directory.

        Returns:
            The path to the save directory.
        """
        return self._save_dir

    @property
    def name(self) -> str:
        """Gets the experiment name.

        Returns:
             The experiment name if the experiment exists, else the name specified in the constructor.
        """
        if self._experiment is None:
            return self._name

        return self.experiment.name

    @property
    def version(self) -> int:
        """Gets the experiment version.

        Returns:
             The experiment version if the experiment exists, else the next version.
        """
        if self._experiment is None:
            return self._version

        return self.experiment.version

    # Test tube experiments are not pickleable, so we need to override a few
    # methods to get DDP working. See
    # https://docs.python.org/3/library/pickle.html#handling-stateful-objects
    # for more info.
    def __getstate__(self) -> Dict[Any, Any]:
        state = self.__dict__.copy()
        state["_experiment"] = self.experiment.get_meta_copy()
        return state

    def __setstate__(self, state: Dict[Any, Any]):
        self._experiment = state["_experiment"].get_non_ddp_exp()
        del state["_experiment"]
        self.__dict__.update(state)
