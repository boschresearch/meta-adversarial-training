# This is source code modified from
# https://github.com/keras-team/keras/blob/7a39b6c62d43c25472b2c2476bd2a8983ae4f682/examples/cifar10_resnet.py
# which is licensed under MIT License (MIT)
#
# COPYRIGHT
#
# All contributions by François Chollet:
# Copyright (c) 2015 - 2019, François Chollet.
# All rights reserved.
#
# All contributions by Google:
# Copyright (c) 2015 - 2019, Google, Inc.
# All rights reserved.
#
# All contributions by Microsoft:
# Copyright (c) 2017 - 2019, Microsoft, Inc.
# All rights reserved.
#
# All other contributions:
# Copyright (c) 2015 - 2019, the respective contributors.
# All rights reserved.
#
# Each contributor holds copyright over their respective contributions.
# The project versioning (Git) records all such contribution source information.
#
# LICENSE
#
# The MIT License (MIT)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""ResNet definitions
ResNet v1
[a] Deep Residual Learning for Image Recognition
https://arxiv.org/pdf/1512.03385.pdf

ResNet v2
[b] Identity Mappings in Deep Residual Networks
https://arxiv.org/pdf/1603.05027.pdf
"""
from __future__ import print_function

import tensorflow.keras as keras
from tensorflow.keras.layers import (
    AveragePooling2D,
    Input,
    Flatten,
    Dense,
    Activation,
    BatchNormalization,
    LayerNormalization,
)
from .convolutional import Conv2D
from .groupnormalization import GroupNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD


# Model parameter
# ----------------------------------------------------------------------------
#           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
#           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
# ----------------------------------------------------------------------------
# ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
# ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
# ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
# ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
# ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
# ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
# ---------------------------------------------------------------------------
n = 3
width = 1

# Model version
# Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
version = 1

# Computed depth from supplied model parameter n
if version == 1:
    depth = n * 6 + 2
elif version == 2:
    depth = n * 9 + 2


def normalize(x, normalization, weight_decay=1e-4):
    if normalization == "batch":
        x = BatchNormalization()(x)
    elif normalization == "layer":
        x = LayerNormalization()(x)
    elif normalization == "group":
        x = GroupNormalization(groups=8)(x)
    elif normalization != "none":
        raise NotImplementedError()
    return x


def resnet_layer(
    inputs,
    num_filters=16,
    kernel_size=3,
    strides=1,
    activation="relu",
    normalization="batch",
    weight_standardization=False,
    conv_first=True,
    weight_decay=1e-4,
):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            activation-bn-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(
        num_filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        standardization=weight_standardization,
        kernel_initializer="he_normal",
        kernel_regularizer=l2(weight_decay),
    )

    x = inputs
    if conv_first:
        x = conv(x)
        x = normalize(x, normalization, weight_decay=weight_decay)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        x = normalize(x, normalization, weight_decay=weight_decay)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(
    input_shape,
    depth,
    width=1,
    batch_size=None,
    normalization="batch",
    activation="relu",
    num_classes=10,
    logits=False,
    n_stacks=3,
    weight_decay=1e-4,
    weight_standardization=True,
):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError("depth should be 6n+2 (eg 20, 32, 44 in [a])")
    # Start model definition.
    num_filters = 16 * width
    num_res_blocks = int((depth - 2) / 6)

    if batch_size is None:
        inputs = Input(shape=input_shape)
    else:
        inputs = Input(batch_shape=(batch_size,) + input_shape)

    x = resnet_layer(
        inputs=inputs,
        num_filters=num_filters,
        normalization=normalization,
        weight_standardization=weight_standardization,
        weight_decay=weight_decay,
    )
    # Instantiate the stack of residual units
    for stack in range(n_stacks):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(
                inputs=x,
                num_filters=num_filters,
                strides=strides,
                activation=activation,
                normalization=normalization,
                weight_standardization=weight_standardization,
                weight_decay=weight_decay,
            )
            y = resnet_layer(
                inputs=y,
                num_filters=num_filters,
                activation=None,
                normalization=normalization,
                weight_standardization=weight_standardization,
                weight_decay=weight_decay,
            )
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(
                    inputs=x,
                    num_filters=num_filters,
                    kernel_size=1,
                    strides=strides,
                    activation=None,
                    normalization="none",
                    weight_standardization=False,
                    weight_decay=weight_decay,
                )
            if keras.backend.int_shape(x)[-1] != num_filters:
                x = Conv2D(
                    num_filters,
                    kernel_size=(1, 1),
                    padding="same",
                    kernel_initializer="he_normal",
                    kernel_regularizer=l2(weight_decay),
                )(x)
            if keras.backend.int_shape(y)[-1] != num_filters:
                y = Conv2D(
                    num_filters,
                    kernel_size=(1, 1),
                    padding="same",
                    kernel_initializer="he_normal",
                    kernel_regularizer=l2(weight_decay),
                )(y)
            x = keras.layers.add([x, y])
            if activation is not None:
                x = Activation(activation)(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    y = Dense(num_classes, activation="linear", kernel_initializer="he_normal")(y)
    if logits:
        outputs = y
    else:
        outputs = Activation("softmax")(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_v2(input_shape, depth, width=1, num_classes=10, normalization="batch"):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError("depth should be 9n+2 (eg 56 or 110 in [b])")
    # Start model definition.
    num_filters_in = 16 * width
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs, num_filters=num_filters_in, conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = "relu"
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    normalization = None
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2  # downsample

            # bottleneck residual unit
            y = resnet_layer(
                inputs=x,
                num_filters=num_filters_in,
                kernel_size=1,
                strides=strides,
                activation=activation,
                normalization=normalization,
                conv_first=False,
            )
            y = resnet_layer(inputs=y, num_filters=num_filters_in, conv_first=False)
            y = resnet_layer(
                inputs=y, num_filters=num_filters_out, kernel_size=1, conv_first=False
            )
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(
                    inputs=x,
                    num_filters=num_filters_out,
                    kernel_size=1,
                    strides=strides,
                    activation=None,
                    normalization="none",
                )
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes, activation="softmax", kernel_initializer="he_normal")(
        y
    )

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_model(
    initial_weights=None,
    input_shape=(32, 32, 3),
    num_classes=10,
    depth=20,
    width=4,
    n_stacks=3,
    normalization="batch",
    activation="relu",
    weight_decay=1e-4,
    trainable=True,
    logits=False,
):
    model = resnet_v1(
        input_shape=input_shape,
        depth=depth,
        width=width,
        num_classes=num_classes,
        activation=activation,
        normalization=normalization,
        logits=logits,
        n_stacks=n_stacks,
        weight_decay=weight_decay,
    )
    model.trainable = trainable
    opt = SGD(
        lr=0.01,
        momentum=0.9,
        decay=0.0,  # LR will be set by LR scheduler
        nesterov=False,
    )
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    if initial_weights:
        try:
            model.load_weights(initial_weights)
        except ValueError:
            model_for_loading = Model(
                inputs=model.inputs,
                outputs=[Activation("softmax")(model(model.inputs[0]))],
            )

            model_for_loading.load_weights(initial_weights)

    return model
