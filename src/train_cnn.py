from __future__ import annotations

import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from attr import dataclass
from comet_ml import Experiment
from tensorflow.keras.datasets.cifar10 import load_data

from src.resnet import get_resnet_model
from src.types import TFHistory

experiment = Experiment(
    auto_metric_logging=True,
    auto_param_logging=True,
    auto_histogram_weight_logging=True,
    auto_histogram_gradient_logging=True,
    auto_histogram_activation_logging=True,
    project_name="cnn-circles",
    workspace="ahuhn",
)

CIFAR10_CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


@dataclass
class ImageDataSet:
    images: List[Any]
    labels: List[Any]


@dataclass
class ImageData:
    train: ImageDataSet
    test: ImageDataSet


def _get_data() -> ImageData:
    (x_train, y_train), (x_test, y_test) = load_data()

    return ImageData(
        train=ImageDataSet(x_train, y_train), test=ImageDataSet(x_test, y_test)
    )


def train(filter_config: List[Dict[str, Any]]) -> TFHistory:
    input_data = _get_data()

    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1000,
        decay_rate=0.95,
        staircase=True,
    )

    model = get_resnet_model(
        input_data.train.images[0].shape,
        class_count=len(CIFAR10_CLASS_NAMES),
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        input_data.train.images,
        input_data.train.labels,
        epochs=100,
        validation_data=(input_data.test.images, input_data.test.labels),
    )
    print(model.summary())
    print(history.history)

    # plot_history(history)

    test_loss, test_acc = model.evaluate(
        input_data.test.images, input_data.test.labels, verbose=2
    )
    print(test_acc)

    return history


def plot_history(history: TFHistory) -> None:
    plt.plot(history.history["accuracy"], label="train_accuracy")
    plt.plot(history.history["val_accuracy"], label="validation_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim([0.5, 1])
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    filter_config = [
        {
            "filter_shape_name": "square",
            "filter_dims": (2, 2),
            "filter_count": 8,
        },
        {
            "filter_shape_name": "square",
            "filter_dims": (3, 3),
            "filter_count": 16,
        },
        {
            "filter_shape_name": "circle",
            "filter_dims": (3, 3),
            "filter_count": 16,
        },
        {
            "filter_shape_name": "circle",
            "filter_dims": (4, 4),
            "filter_count": 16,
        },
        {
            "filter_shape_name": "vertical_line",  # TODO: Not sure if this is horizontal or vertical
            "filter_dims": (5, 2),
            "filter_count": 8,
        },
        {
            "filter_shape_name": "horizontal_line",  # TODO: Not sure if this is horizontal or vertical
            "filter_dims": (2, 5),
            "filter_count": 8,
        },
        {
            "filter_shape_name": "vertical_line",  # TODO: Not sure if this is horizontal or vertical
            "filter_dims": (5, 1),
            "filter_count": 4,
        },
        {
            "filter_shape_name": "horizontal_line",  # TODO: Not sure if this is horizontal or vertical
            "filter_dims": (1, 5),
            "filter_count": 4,
        },
        # {
        #     "filter_shape_name": "square",
        #     "filter_dims": (3, 3),
        #     "filter_count": 64,
        # },
    ]

    history = train(filter_config)
    # plot_history(history)
