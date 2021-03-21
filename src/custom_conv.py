from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from attr import dataclass
from tensorflow.keras import layers, regularizers
from tensorflow.keras.constraints import Constraint

from custom_types import TFConstantMask, TFTensor


class KernelDistributionType(Enum):
    all_squares = "all_squares"
    mixed = "mixed"


@dataclass
class KernelConfig:
    kernel_shape: KernelShape
    kernel_size: int
    kernel_count: int

    @property
    def name(self) -> str:
        return f"{self.kernel_shape.value}_{self.kernel_size}"


class KernelShape(Enum):
    square = "square"
    circle = "circle"
    vertical_line = "vertical_line"
    horizontal_line = "horizontal_line"


def get_mixed_kernel_configs(
    total_kernel_count: int, approximate_kernel_size: int
) -> List[KernelConfig]:
    config_count = 8
    return [
        KernelConfig(
            kernel_shape=KernelShape.square,
            kernel_size=approximate_kernel_size,
            kernel_count=int(total_kernel_count / config_count),
        ),
        KernelConfig(
            kernel_shape=KernelShape.circle,
            kernel_size=approximate_kernel_size,
            kernel_count=int(total_kernel_count / config_count),
        ),
        KernelConfig(
            kernel_shape=KernelShape.horizontal_line,
            kernel_size=approximate_kernel_size + 2,
            kernel_count=int(total_kernel_count / config_count),
        ),
        KernelConfig(
            kernel_shape=KernelShape.vertical_line,
            kernel_size=approximate_kernel_size + 2,
            kernel_count=int(total_kernel_count / config_count),
        ),
        KernelConfig(
            kernel_shape=KernelShape.horizontal_line,
            kernel_size=approximate_kernel_size,
            kernel_count=int(total_kernel_count / config_count),
        ),
        KernelConfig(
            kernel_shape=KernelShape.vertical_line,
            kernel_size=approximate_kernel_size,
            kernel_count=int(total_kernel_count / config_count),
        ),
        KernelConfig(
            kernel_shape=KernelShape.square,
            kernel_size=approximate_kernel_size - 1,
            kernel_count=int(total_kernel_count / config_count),
        ),
        KernelConfig(
            kernel_shape=KernelShape.circle,
            kernel_size=approximate_kernel_size + 1,
            kernel_count=int(total_kernel_count / config_count),
        ),
    ]


def get_kernel_size_from_config(kernel_config: KernelConfig) -> Tuple[int, int]:
    if kernel_config.kernel_shape == KernelShape.horizontal_line:
        return (kernel_config.kernel_size, 1)
    elif kernel_config.kernel_shape == KernelShape.vertical_line:
        return (1, kernel_config.kernel_size)
    return (kernel_config.kernel_size, kernel_config.kernel_size)


def get_custom_conv(
    x: TFTensor,
    total_kernel_count: int,
    approximate_kernel_size: int,
    stride: int,
    name: str,
    kernel_distribution_type: KernelDistributionType,
    l2_regularization: float = 0.0001,
) -> TFTensor:
    if (
        kernel_distribution_type == KernelDistributionType.all_squares
        or approximate_kernel_size == 1
    ):
        return layers.Conv2D(
            total_kernel_count,
            approximate_kernel_size,
            strides=stride,
            name=name,
            padding="same",
            kernel_regularizer=regularizers.L2(l2=l2_regularization),
            bias_regularizer=regularizers.L2(l2=l2_regularization),
        )(x)
    elif kernel_distribution_type == KernelDistributionType.mixed:
        kernel_configs = get_mixed_kernel_configs(
            total_kernel_count, approximate_kernel_size
        )
        parallel_convs = [
            layers.Conv2D(
                kernel_config.kernel_count,
                get_kernel_size_from_config(kernel_config),
                strides=stride,
                name=name + "_" + kernel_config.name,
                padding="same",
                kernel_constraint=KernelShapeConstraint(
                    kernel_shape=kernel_config.kernel_shape.value,
                    kernel_size=kernel_config.kernel_size,
                ),
                kernel_regularizer=regularizers.L2(l2=l2_regularization),
                bias_regularizer=regularizers.L2(l2=l2_regularization),
            )(x)
            for kernel_config in kernel_configs
        ]
        x = layers.Add(name=name + "_add")(parallel_convs)
        return x
    raise Exception


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
