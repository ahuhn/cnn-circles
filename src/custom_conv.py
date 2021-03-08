from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

import numpy as np
import tensorflow as tf
from attr import dataclass
from tensorflow.keras import layers
from tensorflow.keras.constraints import Constraint

from custom_types import TFConstantMask, TFTensor


def get_custom_conv(
    x: TFTensor,
    kernel_count: int,
    kernel_size: int,
    stride: int,
    name: str,
) -> TFTensor:
    kernel_constraint = KernelShapeConstraint(
        kernel_shape="square", kernel_size=kernel_size
    )
    x = layers.Conv2D(
        kernel_count,
        kernel_size,
        strides=stride,
        name=name,
        padding="same",
        kernel_constraint=kernel_constraint,
    )(x)
    return x


class KernelShapeConstraint(Constraint):
    """
    Adjusts the shape of the filter by multiplying some weights by 0
    """

    def __init__(self, kernel_shape: str, kernel_size: int) -> None:
        self.kernel_shape = kernel_shape
        self.kernel_size = kernel_size
        self.mask = _get_mask(kernel_shape, kernel_size)

    def __call__(self, w: Any) -> Any:
        if self.mask is None:
            return w
        return tf.multiply(self.mask, w)

    def get_config(self) -> Dict[str, Any]:
        return {"kernel_shape": self.kernel_shape, "kernel_size": self.kernel_size}


def _get_mask(kernel_shape: str, kernel_size: int) -> Optional[TFConstantMask]:
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
    if kernel_shape == "circle":
        mask = np.ones(
            [
                kernel_size,
                kernel_size,
                1,
                1,
            ],
            dtype=np.float32,
        )
        mask[0][0] = 0
        mask[0][-1] = 0
        mask[-1][0] = 0
        mask[-1][-1] = 0
        if kernel_size > 5:
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
