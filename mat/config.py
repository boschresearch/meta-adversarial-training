# Copyright (c) 2020 Robert Bosch GmbH
# All rights reserved.
#
# This source code is licensed under the AGPL-3.0 license found in the
# LICENSE file in the root directory of this source tree
"""This module handles configuration of experiments and training."""

import os
from dataclasses import dataclass
from dataclasses import asdict
from typing import List
from uuid import uuid4
import datetime

import yaml


@dataclass
class BaseConfig:
    """
    Base Configuration.

    Args:
        base_result_dir: Base directory where results are stored in a folder `unique_name`.
        unique_name: A unique name for the experiment. If none, it's generated.
            When instantiated via `from_yaml`, it can also be a formatting string containing config keys and the special
            keys `id` and `date`.
        data_dir: Path to the data directory
    """

    base_result_dir: str
    unique_name: str
    data_dir: str

    def __post_init__(self):
        self.resolve_unique_name()

    @classmethod
    def from_yaml(cls, path: str):
        """
        Initialize configuration from file.

        Args:
            path: Path to file containing the configuration
        """
        with open(path, "r") as config_file:
            config = yaml.load(config_file, Loader=yaml.FullLoader)
            if isinstance(config, dict):
                return cls(**config)
            return config

    def to_yaml(self, path: str):
        """
        Write configuration to file.

        Args:
            path: Path to the file that will be written
        """
        with open(path, "w") as config_file:
            yaml.dump(self, config_file, default_flow_style=False)

    def resolve_unique_name(self):
        uuid = str(uuid4())[:8]
        date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        format_config = asdict(self)
        format_config.update({"id": uuid, "date": date})

        def stringify(config: dict):
            """Recursively makes all entries printable.

            It resolves the special cases where the dict value
                - is a string containing os.sep
                - is a list
                - is a dict
            """
            for key, value in config.items():
                if isinstance(value, str):
                    config[key] = value.replace(os.sep, "_")
                elif isinstance(value, list):
                    config[key] = "_".join(map(str, value))
                elif isinstance(value, dict):
                    config[key] = stringify(value)
            return config

        format_config = stringify(format_config)
        self.unique_name = self.unique_name.format(**format_config)

        return self


@dataclass
class OptimizerConfig:
    """Inner loop optimizer configuration.

    `min_step_size` and `max_step_size` specify the range of the
    log uniform distribution, where the step sizes for individual
    meta patches are drawn from.

    Args:
        min_step_size: Minimal step size for FGSM steps.
        max_step_size: Maximimal step size for FGSM steps.
        n_iterations: Number of steps in I-FGSM.
        targeted_attack: Whether the I-FGSM attack is targeted.
    """

    min_step_size: float
    max_step_size: float
    n_iterations: int
    targeted_attack: bool = True


@dataclass
class PatchTrainingConfig(BaseConfig):
    """Configuration for MAT training.

    Args:
        patch_shape: Shape of patch
        batch_size: Batch size.
        optimizer: Adversarial Optimizer (I-FGSM) configuration.
        initial_lr: Initial learning rate of model.
        label_smoothing: Label smoothing.
        n_epochs: Number of epochs.
        n_patch_trials: Number of patches tested on an image before starting PGD.
        n_patches: Number of patches to be meta-learned.
        patch_lr: Patch learning rate.
        targeted_attack: Whether a targeted attack is conducted.
        patch_initialization: how the meta-patches are intialized (random or data).
        seed: Seed used for tensorflow and numpy random number generator.
    """

    patch_shape: List[int]
    batch_size: int
    optimizer: OptimizerConfig
    initial_lr: float
    label_smoothing: float
    n_epochs: int
    n_patch_trials: int
    n_patches: int
    patch_lr: float
    patch_initialization: str = "data"
    seed: int = 0
