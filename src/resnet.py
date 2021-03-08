from __future__ import annotations

# from tensorflow.keras import datasets, layers, models, constraints, Input, Sequential, regularizers
from tensorflow.keras import Model
from tensorflow.keras import backend as keras_backend
from tensorflow.keras import layers

from custom_conv import get_custom_conv
from custom_types import KerasModel, TFShape, TFTensor


def get_resnet_model(input_shape: TFShape, class_count: int) -> KerasModel:
    img_input = layers.Input(shape=input_shape)
    bn_axis = 3 if keras_backend.image_data_format() == "channels_last" else 1

    x = get_custom_conv(
        img_input, kernel_count=64, kernel_size=7, stride=2, name="conv1_conv"
    )

    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name="conv1_bn")(x)
    x = layers.Activation("relu", name="conv1_relu")(x)

    # x = stack1(x, kernel_count=64, block_count=3, stride1=1, name="conv2")
    # x = stack1(x, kernel_count=128, block_count=4, name="conv3")
    # x = stack1(x, kernel_count=256, block_count=6, name="conv4")
    # x = stack1(x, kernel_count=512, block_count=3, name="conv5")
    x = stack1(x, kernel_count=32, block_count=3, stride1=1, name="conv2")
    x = stack1(x, kernel_count=64, block_count=4, name="conv3")
    x = stack1(x, kernel_count=128, block_count=6, name="conv4")
    x = stack1(x, kernel_count=256, block_count=3, name="conv5")

    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = layers.Dense(class_count, activation="softmax", name="predictions")(x)

    model = Model(img_input, x, name="resnet50")
    return model


def stack1(
    x: TFTensor,
    kernel_count: int,
    block_count: int,
    name: str,
    stride1: int = 2,
) -> TFTensor:
    """A set of stacked residual blocks.
    Arguments:
      x: input tensor.
      filters: integer, filters of the bottleneck layer in a block.
      blocks: integer, blocks in the stacked blocks.
      stride1: default 2, stride of the first layer in the first block.
      name: string, stack label.
    Returns:
      Output tensor for the stacked blocks.
    """
    x = block1(x, kernel_count, stride=stride1, name=name + "_block1")
    for i in range(2, block_count + 1):
        x = block1(x, kernel_count, conv_shortcut=False, name=name + "_block" + str(i))
    return x


def block1(
    x: TFTensor,
    kernel_count: int,
    name: str,
    kernel_size: int = 3,
    stride: int = 1,
    conv_shortcut: bool = True,
) -> TFTensor:
    """A residual block.
    Arguments:
      x: input tensor.
      filters: integer, filters of the bottleneck layer.
      kernel_size: default 3, kernel size of the bottleneck layer.
      stride: default 1, stride of the first layer.
      conv_shortcut: default True, use convolution shortcut if True,
          otherwise identity shortcut.
      name: string, block label.
    Returns:
      Output tensor for the residual block.
    """
    bn_axis = 3 if keras_backend.image_data_format() == "channels_last" else 1

    if conv_shortcut:
        shortcut = get_custom_conv(
            x,
            kernel_count=4 * kernel_count,
            kernel_size=1,
            stride=stride,
            name=name + "_0_conv",
        )
        shortcut = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + "_0_bn"
        )(shortcut)
    else:
        shortcut = x

    x = get_custom_conv(
        x,
        kernel_count=kernel_count,
        kernel_size=1,
        stride=stride,
        name=name + "_1_conv",
    )
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_1_bn")(
        x
    )
    x = layers.Activation("relu", name=name + "_1_relu")(x)

    x = get_custom_conv(
        x,
        kernel_count=kernel_count,
        kernel_size=kernel_size,
        stride=1,
        name=name + "_2_conv",
    )
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_2_bn")(
        x
    )
    x = layers.Activation("relu", name=name + "_2_relu")(x)

    x = get_custom_conv(
        x, kernel_count=4 * kernel_count, kernel_size=1, stride=1, name=name + "_3_conv"
    )
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_3_bn")(
        x
    )

    x = layers.Add(name=name + "_add")([shortcut, x])
    x = layers.Activation("relu", name=name + "_out")(x)
    return x
