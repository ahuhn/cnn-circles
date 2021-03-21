""" train_cnn.py

    Skipping isort here so comet_ml is imported before tensorflow
    isort:skip_file
"""

from __future__ import annotations


from comet_ml import Experiment
import matplotlib.pyplot as plt
import tensorflow as tf

from cifar10_data import get_cifar10_data, CIFAR10_CLASS_NAMES
from custom_types import TFHistory
from resnet import get_resnet_model
from custom_conv import KernelDistributionType


experiment = Experiment(
    auto_metric_logging=True,
    auto_param_logging=True,
    auto_histogram_weight_logging=True,
    auto_histogram_gradient_logging=True,
    auto_histogram_activation_logging=True,
    project_name="cnn-circles",
    workspace="ahuhn",
)


def train() -> TFHistory:
    input_data = get_cifar10_data()

    initial_learning_rate = 0.01
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate,
    #     decay_steps=500,
    #     decay_rate=0.9,
    #     staircase=True,
    # )

    model = get_resnet_model(
        input_data.train.images[0].shape,
        class_count=len(CIFAR10_CLASS_NAMES),
        kernel_distribution_type=KernelDistributionType.all_squares,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        input_data.train.images,
        input_data.train.labels,
        epochs=20,
        validation_data=(input_data.test.images, input_data.test.labels),
    )
    model.save("trained_models/resnet_squares")
    # print(model.summary())
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
    train()
