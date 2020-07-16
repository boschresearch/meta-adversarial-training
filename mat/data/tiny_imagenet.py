# This is the source code modified from
# https://github.com/uds-lsv/evaluating-logit-pairing-methods/blob/201d3c05b59bb92df267e3c0213a50081988e1bb/tiny_imagenet/datasets/tiny_imagenet_input.py
#
# Copyright 2018 Google Inc. All Rights Reserved.
# Modifications copyright (C) 2018, Maksym Andriushchenko <m.andriushchenko@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# A copy of the Apache-2.0 license can also be found in the 3rd-party-licenses.txt
# file in the root directory of this source tree.
# ==============================================================================

"""Tiny imagenet input."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf


def tiny_imagenet_parser(value, image_size, add_augmentation, one_hot, label_smooth):
    """
    tf.Records parser for tiny ImageNet

    Parameters
    ----------
      value : tf.Tensor
        A scalar string tensor ecoding a single example
      image_size : int
        size of the image.
      add_augmentation : bool
        if True then do training preprocessing (which includes
        random cropping), otherwise do no preprocessing.
      one_hot : bool
        if True return the labels one-hot encoded
      label_smooth: float
        The degree of label smoothing applied

    Returns
    -------
      image : tf.Tensor
        tensor with the image.
      label: tf.Tensor
        tensor with true label of the image.
    """
    keys_to_features = {
        "image/encoded": tf.io.FixedLenFeature((), tf.string, ""),
        "label/tiny_imagenet": tf.io.FixedLenFeature([], tf.int64, -1),
    }

    parsed = tf.io.parse_single_example(value, keys_to_features)

    image_buffer = tf.reshape(parsed["image/encoded"], shape=[])
    image = tf.image.decode_image(image_buffer, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # Crop image
    if add_augmentation:
        bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=tf.constant(
                [0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4]
            ),
            min_object_covered=0.5,
            aspect_ratio_range=[0.75, 1.33],
            area_range=[0.5, 1.0],
            max_attempts=20,
            use_image_if_no_bounding_boxes=True,
        )
        image = tf.slice(image, bbox_begin, bbox_size)
        # Data augmentation
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_saturation(image, 0.5, 2.0)

    # resize image
    image = tf.compat.v1.image.resize_bicubic([image], [image_size, image_size])[0]
    image = tf.reshape(image, [image_size, image_size, 3])
    image = tf.clip_by_value(image, 0, 1)

    # Labels are in [0, 199] range
    label = tf.cast(tf.reshape(parsed["label/tiny_imagenet"], shape=[]), dtype=tf.int32)
    if one_hot:
        label = tf.one_hot(label, depth=200)
        if label_smooth > 0.0:
            label = tf.clip_by_value(label, label_smooth / 199.0, 1.0 - label_smooth)

    return image, label


def load_tiny_imagenet_dataset(
    path,
    batch_size,
    image_size,
    add_train_augmentation=False,
    one_hot=False,
    label_smooth=0.0,
    augment_func=None,
):
    """
    Load Tiny ImageNet as tf.data.Dataset

    Parameters
    ----------
    path : str
        A string specifying the path of the tiny ImageNet dir
    batch_size: size of the minibatch
    image_size: size of the one side of the image. Output images will be
        resized to square shape image_size*image_size
    add_train_augmentation : bool, default False
        if True the train dataset is augmented with cropping, flips
        and saturation
    one_hot : bool
        if True return the labels one-hot encoded
    label_smooth: float
        The degree of label smoothing applied
    augment_func: callable
        Augmentation function that will be applied via dataset.map

    Returns
    -------
        tf.data.Dataset, tf.data.Dataset
            train_dataset, test_dataset as tf.data.Dataset
            Images are normalized to [0,1]

    """
    train_path = os.path.join(path, "train.tfrecord")
    validation_path = os.path.join(path, "validation.tfrecord")

    train_dataset = tf.data.TFRecordDataset(train_path, buffer_size=8 * 1024 * 1024)
    validation_dataset = tf.data.TFRecordDataset(
        validation_path, buffer_size=8 * 1024 * 1024
    )
    label_shape = [batch_size, None] if one_hot else [batch_size]

    def set_shapes(images, labels):
        """Statically set the batch_size dimension."""
        images.set_shape(
            images.get_shape().merge_with(
                tf.TensorShape([batch_size, None, None, None])
            )
        )
        labels.set_shape(labels.get_shape().merge_with(tf.TensorShape(label_shape)))
        return images, labels

    datasets = [train_dataset, validation_dataset]
    add_augmentation = [add_train_augmentation, False]

    for i in range(len(datasets)):
        datasets[i] = datasets[i].apply(
            tf.data.experimental.map_and_batch(
                lambda value: tiny_imagenet_parser(
                    value, image_size, add_augmentation[i], one_hot, label_smooth
                ),
                batch_size=batch_size,
                num_parallel_batches=4,
                drop_remainder=True,
            )
        )

        # Assign static batch size dimension
        datasets[i] = datasets[i].map(set_shapes)
        if augment_func is not None:
            datasets[i] = datasets[i].map(augment_func)
        datasets[i] = datasets[i].prefetch(tf.contrib.data.AUTOTUNE)

    return datasets[0], datasets[1]


