from __future__ import annotations

import tensorflow as tf

from cifar10_data import CIFAR10_CLASS_NAMES, get_cifar10_data
from custom_conv import KernelDistributionType
from custom_types import TFHistory
from resnet import get_resnet_model


def train() -> TFHistory:
    input_data = get_cifar10_data()

    model = get_resnet_model(
        input_data.train.images[0].shape,
        class_count=len(CIFAR10_CLASS_NAMES),
        kernel_distribution_type=KernelDistributionType.all_squares,
        block_count_per_layer=3,
    )
    print(model.summary())

    initial_learning_rate = 0.1
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=10000,
        decay_rate=0.3,
        staircase=True,
    )

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    preprocesser = tf.keras.preprocessing.image.ImageDataGenerator(
        fill_mode="constant",
        cval=0,
        horizontal_flip=True,
        width_shift_range=5,
        height_shift_range=5,
    )
    model.fit(
        preprocesser.flow(
            input_data.train.images, input_data.train.labels, batch_size=128
        ),
        epochs=100,
        validation_data=(input_data.test.images, input_data.test.labels),
    )
    model.save("trained_models/resnet_squares_with_augmentation")
