from __future__ import annotations

from enum import Enum
from typing import Any, Optional

import numpy as np
import tensorflow as tf
from attr import dataclass
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.layers import VersionAwareLayers

from src.types import TFConstantMask, TFTensor

layers = VersionAwareLayers()


def get_custom_conv(
    x: TFTensor,
    kernel_count: int,
    kernel_size: int,
    stride: int,
    name: str,
) -> TFTensor:
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
