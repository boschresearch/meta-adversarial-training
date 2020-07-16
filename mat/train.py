# Copyright (c) 2020 Robert Bosch GmbH
# All rights reserved.
#
# This source code is licensed under the AGPL-3.0 license found in the
# LICENSE file in the root directory of this source tree
""" Meta adversarial training against universal patches on Tiny ImageNet. """

import argparse
import os
import csv

import numpy as np

import tensorflow as tf

from utils.common import IFGSM

from data.tiny_imagenet import load_tiny_imagenet_dataset
from models.resnet import resnet_v1
from utils.common import cosine_anneal_schedule, create_result_dir
from config import PatchTrainingConfig
from patch import PatchSampler, PatchSelector, patch_application

tf.compat.v1.enable_eager_execution()


def meta_adversarial_training(model: tf.keras.Model, config: PatchTrainingConfig):
    """Trains a model meta-adversarial training.

    Args:
        model: A model to be trained.
        config: A training configuration
    """
    patch_lr = config.patch_lr
    initial_lr = config.initial_lr
    n_epochs = config.n_epochs
    batch_size = config.batch_size
    label_smoothing = config.label_smoothing

    n_patches = config.n_patches
    n_patch_trials = config.n_patch_trials

    patch_shape = config.patch_shape

    # Data
    dataset_train, dataset_valid = load_tiny_imagenet_dataset(
        config.data_dir,
        batch_size,
        image_size=64,
        add_train_augmentation=True,
        one_hot=True,
        label_smooth=label_smoothing,
    )

    # Log-file
    csv_file = open(config.result_dir + os.sep + "log.csv", "w")
    csv_writer = csv.writer(csv_file, delimiter=";")
    csv_writer.writerow(["Epoch", "Train Loss", "Train Accuracy", "Valid Accuracy"])

    i_fgsm = IFGSM(
        model,
        patch_application,
        model.loss_functions[0],
        config.optimizer.n_iterations,
        maximize_loss=not config.optimizer.targeted_attack,
    )

    # Objects for sampling new patches and selecting patches to be applied to a batch
    patch_sampler = PatchSampler(
        dataset_train,
        patch_shape,
        patch_initialization=config.patch_initialization,
        targeted=config.optimizer.targeted_attack,
    )
    patch_selector = PatchSelector(model, patch_application, model.loss_functions[0])

    # Generate initial meta patches
    meta_patches, patch_targets = patch_sampler(n_patches)
    # Sample patch-specific step sizes from log-uniform distribution
    step_sizes = tf.convert_to_tensor(
        10
        ** np.random.uniform(
            np.log10(config.optimizer.min_step_size),
            np.log10(config.optimizer.max_step_size),
            (n_patches,),
        ).astype("float32")
    )[:, None, None, None]

    # Adversarial training
    for epoch in range(n_epochs):
        # Set up progress bar and adapt learning rate
        progbar = tf.keras.utils.Progbar(100000 // batch_size)

        lr = cosine_anneal_schedule(epoch, n_epochs, initial_lr)
        tf.keras.backend.set_value(model.optimizer.lr, lr)

        # Train for one epoch (outer minimization)
        for (images, labels) in dataset_train.shuffle(100):
            # Select patches and randomness in adversarial fashion
            # SELECT in Algorithm 1
            patch_ind, randomness = patch_selector(
                images, labels, meta_patches, batch_size, n_patch_trials
            )
            patches = tf.gather(meta_patches, patch_ind)

            # Set patch-specific step sizes
            step_size = tf.gather(step_sizes, patch_ind)

            # Adapt patch to current batch
            target = (
                tf.gather(patch_targets, patch_ind)
                if config.optimizer.targeted_attack
                else labels
            )

            # Run inner maximization
            patches = i_fgsm(patches, images, target, step_size, randomness)

            # Update meta patch using REPTILE
            meta_patches = update_meta_patches(
                patches, patch_ind, meta_patches, patch_lr
            )

            # Add batch-adapted patch to images
            images_with_patch = patch_application(images, patches, randomness)

            # Train model on inputs with patches
            metric_values = model.train_on_batch(images_with_patch, labels)

            progbar.add(1, zip(model.metrics_names, metric_values))

        # Evaluate clean performance of model
        acc_valid = model.evaluate(dataset_valid)[1]

        # Store weights and update log file
        model.save_weights(
            os.path.join(config.result_dir, f"tiny_imagenet_weights_{epoch:03d}.h5")
        )

        csv_writer.writerow(
            map(
                lambda x: "%.3f" % x,
                [
                    epoch,
                    progbar._values["loss"][0] / progbar._values["loss"][1],
                    progbar._values["acc"][0] / progbar._values["acc"][1],
                    acc_valid,
                ],
            )
        )
        csv_file.flush()


def update_meta_patches(
    patches: tf.Tensor, patch_ind: tf.Tensor, meta_patches: tf.Tensor, patch_lr: float
) -> tf.Tensor:
    """Updates the meta patches with refined patches. (REPTILE in Algorithm 1)

    Args:
        patches: Refined patches of shape [B, H_patch, W_patch, 3]
        patch_ind: Indices of `patches` in the meta patches in shape [B, ].
        meta_patches: Meta Patches of shape [P, H_patch, W_patch, 3]
            where P is the total number of meta patches.
        patch_lr: REPTILE learning rate that controls how much
            the meta patches are updated with the refined patches.

    Returns:
        Meta patches updated with the refined patches.
    """
    if patch_lr == 0:
        # Meta patches are not adapted
        return meta_patches

    n_meta_patches = meta_patches.shape[0]
    # The same patch might have been adapted to different images in the
    # batch. Average over these adapted patches originating from the same
    # meta-patch
    aggregated_patch = tf.math.unsorted_segment_mean(
        patches, patch_ind, num_segments=n_meta_patches
    )
    # Some meta-patches might not have been used in this batch and
    # should not be updated
    patch_used_in_batch = tf.cast(
        tf.math.bincount(patch_ind, minlength=n_meta_patches) > 0, "float32"
    )[:, None, None, None]

    # Update remaining meta-patches based on a convex combination of
    # old meta-patch value and value of the batch-adapted patches
    return (
        1 - patch_lr * patch_used_in_batch
    ) * meta_patches + patch_lr * patch_used_in_batch * aggregated_patch


def create_model(initial_lr):
    model = resnet_v1(
        input_shape=(64, 64, 3),
        depth=14,
        width=4,
        num_classes=200,
        activation="relu",
        normalization="group",
        weight_standardization=True,
        logits=False,
        n_stacks=4,
        weight_decay=1e-4,
    )
    opt = tf.keras.optimizers.SGD(
        lr=initial_lr, momentum=0.9, decay=0.0, nesterov=False
    )
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    return model


def main():
    parser = argparse.ArgumentParser(description="Adversarial training.")
    parser.add_argument("config", type=str)

    args = parser.parse_args()
    config = PatchTrainingConfig.from_yaml(args.config)

    config.result_dir = create_result_dir(config)

    tf.compat.v1.set_random_seed(config.seed)
    np.random.seed(config.seed)

    model = create_model(config.initial_lr)

    meta_adversarial_training(model, config)


if __name__ == "__main__":
    main()
