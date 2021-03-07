from __future__ import annotations

from enum import Enum
from typing import Any, Optional

import numpy as np
import tensorflow as tf
from attr import dataclass
from tensorflow.python.keras import backend
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.constraints import Constraint
from tensorflow.python.keras.datasets.cifar import load_batch
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers import VersionAwareLayers
from tensorflow.python.keras.utils import data_utils, layer_utils
from tensorflow.python.lib.io import file_io
from tensorflow.python.util.tf_export import keras_export

from src.types import KerasModel, TFConstantMask, TFShape

layers = VersionAwareLayers()


def get_custom_conv(
    x: KerasModel,
    kernel_count: int,
    kernel_size: int,
    stride: int,
    name: str,
) -> KerasModel:
    kernel_config = KernelConfig(
        kernel_shape=KernelShape.square,
        kernel_size=kernel_size,
    )
    kernel_constraint = KernelShapeConstraint(kernel_config)
    x = layers.Conv2D(
        kernel_count,
        kernel_size,
        strides=stride,
        name=name,
        kernel_constraint=kernel_constraint,
    )(x)
    return x


@dataclass
class KernelConfig:
    kernel_shape: KernelShape
    kernel_size: int


class KernelShape(Enum):
    square = "square"
    circle = "circle"
    vertical_line = "vertical_line"
    horizontal_line = "horizontal_line"


class KernelShapeConstraint(Constraint):
    """
    Adjusts the shape of the filter by multiplying some weights by 0
    """

    def __init__(self, kernel_config: KernelConfig) -> None:
        self.kernel_config = kernel_config
        self.mask = _get_mask(kernel_config)

    def __call__(self, w: Any) -> Any:
        if self.mask is None:
            return w
        return tf.multiply(self.mask, w)

    def get_config(self) -> KernelConfig:
        return self.kernel_config


def _get_mask(kernel_config: KernelConfig) -> Optional[TFConstantMask]:
    """
    This creates a filter mask.

    For instance,
    ┌───┬───┬───┐
    │ 0 │ 1 │ 0 │
    ├───┼───┼───┤
    │ 1 │ 1 │ 1 │
    ├───┼───┼───┤
    │ 0 │ 1 │ 0 │
    └───┴───┴───┘

    or if it is larger,
    ┌───┬───┬───┬───┬───┬───┐
    │ 0 │ 0 │ 1 │ 1 │ 0 │ 0 │
    ├───┼───┼───┼───┼───┼───┤
    │ 0 │ 1 │ 1 │ 1 │ 1 │ 0 │
    ├───┼───┼───┼───┼───┼───┤
    │ 1 │ 1 │ 1 │ 1 │ 1 │ 1 │
    ├───┼───┼───┼───┼───┼───┤
    │ 1 │ 1 │ 1 │ 1 │ 1 │ 1 │
    ├───┼───┼───┼───┼───┼───┤
    │ 0 │ 1 │ 1 │ 1 │ 1 │ 0 │
    ├───┼───┼───┼───┼───┼───┤
    │ 0 │ 0 │ 1 │ 1 │ 0 │ 0 │
    └───┴───┴───┴───┴───┴───┘
    """
    if kernel_config.kernel_shape == KernelShape.circle:
        mask = np.ones(
            [
                kernel_config.kernel_size,
                kernel_config.kernel_size,
                1,
                1,
            ],
            dtype=np.float32,
        )
        mask[0][0] = 0
        mask[0][-1] = 0
        mask[-1][0] = 0
        mask[-1][-1] = 0
        if kernel_config.kernel_size > 5:
            mask[0][1] = 0
            mask[1][0] = 0
            mask[0][-2] = 0
            mask[1][-1] = 0
            mask[-1][1] = 0
            mask[-2][0] = 0
            mask[-1][-2] = 0
            mask[-2][-1] = 0
        return tf.constant(mask)
    return None


# class ConvLayerWithFilters(layers.Layer):
#     def __init__(self, hyperparams):
#         super(ConvLayerWithFilters, self).__init__()
#         self.hyperparams = hyperparams
#         print("trainable", self.trainable)

#     def build(self, input_shape):
#         self.sublayer_list = []
#         print("input_shape", input_shape)

#         channels_in = input_shape[-1]

#         # Really basic attempt: just one sublayer with 64 3x3 kernels
#         # conv = layers.Conv2D(64, (3, 3), padding="same", activation='relu', input_shape=(32, 32, channels_in))
#         # self.sublayer_list.append(conv)

#         # Split the 64 3x3 kernels into 8 sub-layers. Slower, but similar validation accuracy as the one above
#         # for i in range(0, 8):
#         #     conv = layers.Conv2D(8, (3, 3), padding="same", activation='relu', input_shape=(32, 32, channels_in))
#         #     self.sublayer_list.append(conv)

#         for i, filter_out in enumerate(self.hyperparams["filter_config"]):
#             # print(i)
#             # print(filter_out)
#             # print(input_shape)
#             channels_out = filter_out["kernel_count"]
#             constraint = FilterShapeConstraint(
#                 filter_out["filter_shape_name"],
#                 filter_out["filter_dims"],
#                 channels_in,
#                 channels_out,
#             )
#             conv = layers.Conv2D(
#                 channels_out,
#                 filter_out["filter_dims"],
#                 padding="same",
#                 activation="relu",
#                 kernel_constraint=constraint,
#                 input_shape=input_shape,
#                 kernel_regularizer=regularizers.L2(l2=self.hyperparams["l2"]),
#             )
#             # # pool = layers.MaxPooling2D((2, 2))
#             # # normalize = layers.BatchNormalization()
#             # sub_model = Sequential()
#             # sub_model.add(conv)
#             # # sub_model.add(pool)
#             # # sub_model.add(normalize)
#             # sub_model.build(input_shape)
#             self.sublayer_list.append(conv)

#             # old stuff:
#             # conv = layers.Conv2D(64, (3, 3), padding="same", activation='relu', input_shape=(32, 32, 3))
#             # pool = layers.MaxPooling2D((2, 2))
#             # normalize = None # layers.BatchNormalization()
#             # self.sublayer_list.append((conv, pool, normalize))

#         # print(len(self.sublayer_list[0][0].get_weights()))
#         # print(len(self.sublayer_list[0][0].trainable_variables))
#         # print(len(self.sublayer_list[0][0].non_trainable_variables))
#         super(ConvLayerWithFilters, self).build(input_shape)

#     # @tf.function()
#     def call(self, input_data):
#         # print(len(self.get_weights()))
#         # print(len(self.trainable_variables))
#         # print(len(self.non_trainable_variables))
#         output_list = []
#         for conv in self.sublayer_list:
#             output_list.append(conv(input_data))

#         if not output_list:
#             return input_data
#         elif len(output_list) == 1:
#             return output_list[0]
#         else:
#             return layers.Concatenate()(output_list)

#     def compute_output_shape(self, input_shape):
#         return input_shape

#     def get_config(self):
#         return {"filter_config": filter_config}
