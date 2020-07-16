# Copyright (c) 2020 Robert Bosch GmbH
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree
"""This module contains patch initialization methods."""

from typing import Iterator

import tensorflow as tf


def initialize_patch_randomly(shape: list):
    """Choose initialization for patch based on sampling from [0, 1].

    Args:
        shape: The patch shape in [H_patch, W_patch, 3].

    Returns:
        Patch of shape specified in `shape`.
    """
    return tf.random_uniform(shape, minval=0, maxval=1, dtype="float32")


def initialize_patch_from_data(
    data_iterator: Iterator,
    shape: list,
    mode: str = "resize",
    target_label: int = None,
):
    """Choose initialization for patch based on actual data points.

    Args:
        data_iterator: An iterator that yields images and labels.
        shape: The patch shape in [H_patch, W_patch, 3].
        mode: How to get from image shape to patch shape.
            Either `resize` or `crop`.
        target_label: Target label of the data point the patch is initialized from.
            If not specified a random data point.

    Returns:
        Patch of shape specified in `shape`.
    """
    for (image_batch, label_batch) in data_iterator:
        if target_label is None:
            image = image_batch[0]
        else:
            # Find first image in batch with the correct `target_label`
            target_indices = tf.where(
                tf.equal(tf.argmax(label_batch, axis=-1), target_label)
            )
            if target_indices.shape[0] > 0:
                image = image_batch[target_indices[0, 0]]
            else:
                continue

        # Resize/crop image to patch-size
        if mode == "resize":
            return tf.image.resize(image, shape[:2])
        elif mode == "crop":
            return tf.image.random_crop(image, shape)
        else:
            raise ValueError(f"Unsupported mode {mode}")
