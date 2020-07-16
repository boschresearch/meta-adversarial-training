# Copyright (c) 2020 Robert Bosch GmbH
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree
"""This module contains shared utilities."""

import os
from typing import Union

import numpy as np
import tensorflow as tf

from config import BaseConfig
from patch import patch_application


def cosine_anneal_schedule(
    current_epoch: int, total_epochs: int, base_lr: float
) -> float:
    """Calculates the current learning rate following a cosine annealing.

    Args:
        current_epoch: Current epoch.
        total_epochs: Total number of epochs.
        base_lr: Initial/base learning rate.

    Returns:
        Learning rate for the current epoch.
    """
    cos_inner = np.pi * (current_epoch % total_epochs)
    cos_inner /= total_epochs
    cos_out = np.cos(cos_inner) + 1
    return float(base_lr / 2 * cos_out)


def create_result_dir(config: BaseConfig) -> str:
    """Creates a result directory and saves the current config in this directory.

    Args:
        config: A configuration containing `base_result_dir` and `unique_name`.
            The new directory is created at path base_result_dir/unique_name)

    Returns:
        Path to the created result directory.
    """
    result_dir = os.path.join(config.base_result_dir, config.unique_name)
    os.makedirs(result_dir)
    config.to_yaml(os.path.join(result_dir, "config.yaml"))

    return result_dir


class IFGSM:
    """An Iterative Fast Gradient Sign optimizer.

    Corresponds to I-FGSM in Algorithm 1.

    Args:
        model: A model under attack.
        patch_application_fct: A function that applies patches to images.
            Signature is patch_application_fct(images, patches, randomness).
        loss_fct: A loss function.
            Signature is loss_fct(targets, predictions).
        n_iter: The number of steps along the gradient.
        maximize_loss: Whether the loss is maximized or minimized.
            This corresponds to untargeted vs targeted attack.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        patch_application_fct: callable,
        loss_fct: callable,
        n_iter: int,
        maximize_loss: bool,
    ):
        self.model = model
        self.patch_application_fct = patch_application_fct
        self.loss_fct = loss_fct
        self.n_iter = n_iter
        self.maximize_loss = maximize_loss

    def __call__(
        self,
        patches: tf.Tensor,
        images: tf.Tensor,
        targets: tf.Tensor,
        step_sizes: tf.Tensor,
        randomness: Union[np.array, tf.Tensor] = None,
    ) -> tf.Tensor:
        """Optimizes patches via I-FGSM.

        The patches are applied to the given images with given randomness and then
        optimized via the iterative fast gradient sign method.
        Depending on `self.maximize_loss`, `targets` can either be true labels or
        the targets for a targeted attack.


        Args:
            patches: Patches of shape [B, H_patch, W_patch, 3] that are optimized.
            images: Images of shape [B, H, W, 3] that the patches are applied to.
            targets: Target or true labels of shape [B, ].
            step sizes: Step sizes of attack, i.e. scaling factors of sign(gradients).
                Shape is [B, 1, 1, 1], i.e. every patch has its own step size.
            randomness: Randomness that is used to apply the patches. Shape is [B, ].

        Returns:
            Optimized patches of shape [B, H_patch, W_patch, 3].

        """
        for _ in range(self.n_iter):
            with tf.GradientTape() as tape:
                tape.watch(patches)
                images_with_patches = patch_application(images, patches, randomness)
                loss = self.loss_fct(targets, self.model(images_with_patches))
            gradient = tape.gradient(loss, patches)
            if self.maximize_loss:
                patches = tf.clip_by_value(
                    patches + step_sizes * tf.sign(gradient), 0, 1
                )
            else:
                patches = tf.clip_by_value(
                    patches - step_sizes * tf.sign(gradient), 0, 1
                )

            patches = tf.stop_gradient(patches)

        return patches
