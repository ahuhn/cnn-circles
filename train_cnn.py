from __future__ import annotations

from comet_ml import Experiment

import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models, constraints, Input, Sequential, regularizers
from tensorflow.python.keras.datasets.cifar import load_batch
import matplotlib.pyplot as plt

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
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck',
]

def _get_data():
    num_train_samples = 50000
    path = "/Users/anikahuhn/code/cnn-circles/data/cifar-10-batches-py"

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (
            x_train[(i - 1) * 10000:i * 10000, :, :, :],
            y_train[(i - 1) * 10000:i * 10000],
        ) = load_batch(fpath)

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)

    x_test = x_test.astype(x_train.dtype)
    y_test = y_test.astype(y_train.dtype)

    # Normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    return (x_train, y_train), (x_test, y_test)


def _get_mask(filter_shape_name, filter_dims, channels_in, channels_out):
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
    mask = np.ones([filter_dims[0], filter_dims[1], channels_in, channels_out], dtype=np.float32)
    if filter_shape_name == "square":
        pass
    elif filter_shape_name == "circle":
        mask[0][0] = 0
        mask[0][-1] = 0
        mask[-1][0] = 0
        mask[-1][-1] = 0
        if filter_dims[0] > 5 and filter_dims[1] > 5:
            mask[0][1] = 0
            mask[1][0] = 0
            mask[0][-2] = 0
            mask[1][-1] = 0
            mask[-1][1] = 0
            mask[-2][0] = 0
            mask[-1][-2] = 0
            mask[-2][-1] = 0
    return tf.constant(mask)

class FilterShapeConstraint(constraints.Constraint):
    """
    Adjusts the shape of the filter by multiplying some weights by 0
    """

    def __init__(self, filter_shape_name, filter_dims, channels_in, channels_out):
        self.filter_shape_name = filter_shape_name
        self.filter_dims = filter_dims
        self.mask = _get_mask(filter_shape_name, filter_dims, channels_in, channels_out)

    def __call__(self, w):
        return tf.multiply(self.mask, w)

    def get_config(self):
        return {"filter_shape_name": self.filter_shape_name, "filter_dims": self.filter_dims, "channels_in": channels_in}


class ConvLayerWithFilters(layers.Layer):
    def __init__(self, hyperparams):
        super(ConvLayerWithFilters, self).__init__()
        self.hyperparams = hyperparams
        print("trainable", self.trainable)

    def build(self, input_shape):
        self.sublayer_list = []
        print("input_shape", input_shape)

        channels_in = input_shape[-1]

        # Really basic attempt: just one sublayer with 64 3x3 kernels
        # conv = layers.Conv2D(64, (3, 3), padding="same", activation='relu', input_shape=(32, 32, channels_in))
        # self.sublayer_list.append(conv)

        # Split the 64 3x3 kernels into 8 sub-layers. Slower, but similar validation accuracy as the one above
        # for i in range(0, 8):
        #     conv = layers.Conv2D(8, (3, 3), padding="same", activation='relu', input_shape=(32, 32, channels_in))
        #     self.sublayer_list.append(conv)

        for i, filter_out in enumerate(self.hyperparams["filter_config"]):
            # print(i)
            # print(filter_out)
            # print(input_shape)
            channels_out = filter_out["filter_count"]
            constraint = FilterShapeConstraint(filter_out["filter_shape_name"], filter_out["filter_dims"], channels_in, channels_out)
            conv = layers.Conv2D(
                channels_out,
                filter_out["filter_dims"],
                padding="same",
                activation='relu',
                kernel_constraint=constraint,
                input_shape=input_shape,
                kernel_regularizer=regularizers.L2(l2=self.hyperparams["l2"]),
            )
            # # pool = layers.MaxPooling2D((2, 2))
            # # normalize = layers.BatchNormalization()
            # sub_model = Sequential()
            # sub_model.add(conv)
            # # sub_model.add(pool)
            # # sub_model.add(normalize)
            # sub_model.build(input_shape)
            self.sublayer_list.append(conv)

            # old stuff:
            # conv = layers.Conv2D(64, (3, 3), padding="same", activation='relu', input_shape=(32, 32, 3))
            # pool = layers.MaxPooling2D((2, 2))
            # normalize = None # layers.BatchNormalization()
            # self.sublayer_list.append((conv, pool, normalize))

        # print(len(self.sublayer_list[0][0].get_weights()))
        # print(len(self.sublayer_list[0][0].trainable_variables))
        # print(len(self.sublayer_list[0][0].non_trainable_variables))
        super(ConvLayerWithFilters, self).build(input_shape)

    # @tf.function()
    def call(self, input_data):
        # print(len(self.get_weights()))
        # print(len(self.trainable_variables))
        # print(len(self.non_trainable_variables))
        output_list = []
        for conv in self.sublayer_list:
            output_list.append(conv(input_data))

        if not output_list:
            return input_data
        elif len(output_list) == 1:
            return output_list[0]
        else:
            return layers.Concatenate()(output_list)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {"filter_config": filter_config}


def _build_cnn_model_graph(filter_config):
    model = Sequential()

    # model.add(layers.Conv2D(32, (3, 3), padding="same", activation='relu', input_shape=(32, 32, 3)))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(64, (3, 3), padding="same", activation='relu'))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(64, (3, 3), padding="same", activation='relu'))
    model.add(ConvLayerWithFilters(hyperparams={"filter_config": filter_config, "l2": 0.0003}))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(ConvLayerWithFilters(hyperparams={"filter_config": filter_config, "l2": 0.001}))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(ConvLayerWithFilters(hyperparams={"filter_config": filter_config, "l2": 0.003}))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(ConvLayerWithFilters(hyperparams={"filter_config": filter_config, "l2": 0.01}))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu', kernel_regularizer=regularizers.L2(l2=0.01)))
    model.add(layers.Dense(10, activation='softmax', kernel_regularizer=regularizers.L2(l2=0.01)))
    return model


def train(filter_config):
    (train_images, train_labels), (test_images, test_labels) = _get_data()

    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1000,
        decay_rate=0.95,
        staircase=True,
    )

    model = _build_cnn_model_graph(filter_config)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        train_images,
        train_labels,
        epochs=100,
        validation_data=(test_images, test_labels),
    )
    print(model.summary())
    print(history.history)

    # plot_history(history)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(test_acc)

    return history


def plot_history(history):
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label = 'validation_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':
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
