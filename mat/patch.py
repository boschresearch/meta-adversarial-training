# Copyright (c) 2020 Robert Bosch GmbH
# All rights reserved.
#
# This source code is licensed under the AGPL-3.0 license found in the
# LICENSE file in the root directory of this source tree
"""This module contains patch utilities."""

import copy
from typing import Union, Tuple

import numpy as np
import tensorflow as tf

from utils.patch_initialization import (
    initialize_patch_from_data,
    initialize_patch_randomly,
)


class PatchSampler:
    """Allows sampling patches from a dataset.

    Corresponds to INIT in Algorithm 1.

    Args:
        dataset: A dataset where the patches are sampled from.
        patch_shape: Shape of the batches in form [H, W, 3].
        targeted: If `True` and `patch_initialization=data`, this is done in a targeted manner.
            That is: for every patch, a different target class is selected (round robin) and a patch is
            chosen from the dataset such that the patch's label corresponds to the target class.
        patch_initialization: Initialization mode, can be either `data` or `random`.
            `data`: Sample patches from downscaled/cropped images from `dataset`
            `random`: Initialize patches from random uniform [0, 1].
        n_classes: Number of classes in `dataset`.
    """

    def __init__(
        self,
        dataset: tf.data.Dataset,
        patch_shape: list,
        targeted: bool = True,
        patch_initialization: str = "data",
        n_classes: int = 200,
    ):
        self.dataset = dataset.shuffle(buffer_size=100).repeat()
        self.patch_shape = patch_shape
        self.targeted = targeted
        self.n_classes = n_classes
        self.patch_initialization = patch_initialization
        self.sample_counter = -1

    def __call__(self, n_patches: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """Samples `n_patches` separately from dataset. (INIT in Algorithm 1)

        Args:
            n_patches: Number of patches to sample.

        Returns:
            Patches and the corresponding targets.
        """
        patches = []
        patch_targets = []
        data_iterator = iter(self.dataset)

        for i in range(n_patches):
            self.sample_counter += 1
            target_label = self.sample_counter % self.n_classes

            if self.patch_initialization == "data":
                patch = initialize_patch_from_data(
                    data_iterator,
                    shape=self.patch_shape,
                    target_label=target_label if self.targeted else None,
                )
            elif self.patch_initialization == "random":
                patch = initialize_patch_randomly(self.patch_shape)
            else:
                raise ValueError(
                    f"Unsuported initialization {self.patch_initialization}"
                )

            patches.append(patch.numpy())
            patch_target = np.zeros(self.n_classes)
            patch_target[target_label] = 1
            patch_targets.append(patch_target)
        patches = np.stack(patches).astype("float32")
        patch_targets = np.stack(patch_targets).astype("float32")

        return tf.convert_to_tensor(patches), tf.convert_to_tensor(patch_targets)


class PatchSelector:
    """ Select meta-patches and randomness in an adversarial fashion.

    Corresponds to SELECT in Algorithm 1.

    Args:
        model: A model that
        patch_application_fct: A function that applies patches to images.
            Signature is patch_application_fct(images, patches, randomness).
        loss_fct: A loss function.
            Signature is loss_fct(targets, predictions).
    """

    def __init__(self, model, patch_application_fct, loss_fct):
        self.model = model
        self.patch_application_fct = patch_application_fct
        # We need the loss value for every batch element in order to
        # find the most effective against the respective batch element.
        self.loss_fct = copy.copy(loss_fct)
        self.loss_fct.reduction = "none"

    def __call__(
        self,
        images: tf.Tensor,
        labels: tf.Tensor,
        meta_patches: tf.Tensor,
        n_patches_selected: int,
        n_patch_trials: int,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Select most effective meta-patches and randomness for current batch (SELECT in Algorithm 1).

        In `n_patch_trials` trials, the most effective meta-patch + randomness combinations are determined.

        Args:
            images: Images of the current batch.
            labels: Labels of the current batch.
            meta_patches: The current collection of meta-patches.
            n_patches_selected: How many patches (with replacement) shall be selected.
            n_patch_trials: How many random samples are drawn in the adversarial
                selection.

        Returns:
            List of indices and list of randomness
        """
        # Sample patch and randomness
        patch_ind = tf.convert_to_tensor(
            np.random.choice(meta_patches.shape[0], n_patches_selected).astype("int32")
        )
        randomness = np.random.randint(np.iinfo(np.int32).max, size=n_patches_selected)

        if n_patch_trials > 1:
            # Resample repeatedly and check whether we find a better patch/randomness combination

            patch = tf.gather(meta_patches, patch_ind)
            images_with_patch = self.patch_application_fct(images, patch, randomness)
            loss = self.loss_fct(labels, self.model(images_with_patch))

            for trial_ind in range(1, n_patch_trials):
                patch_ind_new = tf.convert_to_tensor(
                    np.random.choice(meta_patches.shape[0], n_patches_selected).astype(
                        "int32"
                    )
                )
                randomness_new = np.random.randint(
                    np.iinfo(np.int32).max, size=n_patches_selected
                )

                patch = tf.gather(meta_patches, patch_ind_new)
                images_with_patch = self.patch_application_fct(
                    images, patch, randomness_new
                )
                loss_new = self.loss_fct(labels, self.model(images_with_patch))
                loss_increased = loss_new > loss

                randomness = tf.where(loss_increased, randomness_new, randomness)
                patch_ind = tf.where(loss_increased, patch_ind_new, patch_ind)

                loss = tf.maximum(loss, loss_new)

        return patch_ind, randomness


def patch_application(
    images: tf.Tensor, patches: tf.Tensor, randomness: Union[np.array, tf.Tensor] = None
):
    """Applies patches to images at random locations.

    Args:
        images: A batch of images of shape [B, H, W, 3].
        patches: A batch of patches of shape [B, H_patch, W_patch, 3].
        randomness: Optional seeds for randomness. Shape is [B, ]

    Returns:
        Images with patches applied at random locations. Shape is [B, H, W, 3].
    """
    image_height, image_width = images.shape[1:3]
    patch_height, patch_width = patches.shape[1:3]
    height_to_pad = int(image_height - patch_height)
    width_to_pad = int(image_width - patch_width)

    # Generate offsets for random translation
    if randomness is None:
        height_offsets = tf.random.uniform(
            [images.shape[0]], maxval=height_to_pad, dtype=tf.int32
        )
        width_offsets = tf.random.uniform(
            [images.shape[0]], maxval=width_to_pad, dtype=tf.int32
        )
    else:
        assert randomness.shape[0] == images.shape[0]
        randomness = randomness.numpy() if tf.is_tensor(randomness) else randomness
        height_offsets = randomness % height_to_pad
        width_offsets = (randomness // height_to_pad) % width_to_pad

    mesh = tf.meshgrid(
        tf.range(images.shape[0]),
        tf.range(patch_height),
        tf.range(patch_width),
        indexing="ij",
    )
    mesh[1] += height_offsets[:, None, None]
    mesh[2] += width_offsets[:, None, None]
    indices = tf.stack(mesh, axis=-1)
    # Paste patches into the images at random locations
    return tf.tensor_scatter_nd_update(images, indices, patches)
