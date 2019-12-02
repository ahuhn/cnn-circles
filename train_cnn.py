from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from comet_ml import Experiment

import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models, constraints
import matplotlib.pyplot as plt

CIFAR10_CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


def _get_data():
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0
    return (train_images, train_labels), (test_images, test_labels)


def _get_mask(filter_shape_name, filter_dims, channels_in):
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
    mask = np.ones([filter_dims[0], filter_dims[1], channels_in])
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
    return mask


class FilterShapeConstraint(constraints.Constraint):
    """
    Adjusts the shape of the filter by multiplying some weights by 0
    Arguments:
      axis: integer, axis along which to calculate weight norms.
          For instance, in a `Dense` layer the weight matrix
          has shape `(input_dim, output_dim)`,
          set `axis` to `0` to constrain each weight vector
          of length `(input_dim,)`.
          In a `Conv2D` layer with `data_format="channels_last"`,
          the weight tensor has shape
          `(rows, cols, input_depth, output_depth)`,
          set `axis` to `[0, 1, 2]`
          to constrain the weights of each filter tensor of size
          `(rows, cols, input_depth)`.
    """

    def __init__(self, filter_shape_name, filter_dims, channels_in):
        self.filter_shape_name = filter_shape_name
        self.filter_dims = filter_dims
        self.mask = _get_mask(filter_shape_name, filter_dims, channels_in)

    def __call__(self, w):
        return w * self.mask

    def get_config(self):
        return {"filter_shape_name": self.filter_shape_name, "filter_dims": self.filter_dims, "channels_in": channels_in}

def _add_conv_layer(channels_in, filters_out, should_pool):
    conv_list = []

    # Each loop takes in conv_in and adds new channels to conv_out. Each needs to be the same size.
    for i, filter_out in enumerate(filters_out):
        constraint = FilterShapeConstraint(filter_out["filter_shape_name"], filter_out["filter_dims"], channels_in)

        conv = layers.Conv2D(filter_out["filter_count"], filter_out["filter_dims"], activation='relu', kernel_constraint=constraint, input_shape=(32, 32, channels_in))
        if should_pool:
            conv = layers.MaxPooling2D((2, 2))(conv)
        conv_list.append(conv)

    return layers.Concatenate()(conv_list)

def _get_filters_out_from_hyper_params(hyper_params):
    filters_out = []
    if "count_2x2_square" in hyper_params and hyper_params["count_2x2_square"]:
        filters_out.append({"filter_shape_name": "square", "filter_dims": (2, 2), "filter_count": hyper_params["count_2x2_square"]})
    if "count_4x4_circle" in hyper_params and hyper_params["count_4x4_circle"]:
        filters_out.append({"filter_shape_name": "circle", "filter_dims": (4, 4), "filter_count": hyper_params["count_4x4_circle"]})
    if "count_3x3_square" in hyper_params and hyper_params["count_3x3_square"]:
        filters_out.append({"filter_shape_name": "square", "filter_dims": (3, 3), "filter_count": hyper_params["count_3x3_square"]})
    if "count_3x3_circle" in hyper_params and hyper_params["count_3x3_circle"]:
        filters_out.append({"filter_shape_name": "circle", "filter_dims": (3, 3), "filter_count": hyper_params["count_3x3_circle"]})
    if "count_6x6_horizontal_line" in hyper_params and hyper_params["count_6x6_horizontal_line"]:
        filters_out.append({"filter_shape_name": "horizontal_line", "filter_dims": (6, 2), "filter_count": hyper_params["count_6x6_horizontal_line"]})
    if "count_6x6_vertical_line" in hyper_params and hyper_params["count_6x6_vertical_line"]:
        filters_out.append({"filter_shape_name": "vertical_line", "filter_dims": (2, 6), "filter_count": hyper_params["count_6x6_vertical_line"]})
    return filters_out

def _build_cnn_model_graph(hyper_params):
    filters_out = _get_filters_out_from_hyper_params(hyper_params)
    filters_out_count = sum(filter_out["filter_count"] for filter_out in filters_out)
    model = models.Sequential()
    model.add(_add_conv_layer(channels_in=3, filters_out=filters_out, should_pool=True))
    model.add(_add_conv_layer(channels_in=filters_out_count, filters_out=filters_out, should_pool=True))
    model.add(_add_conv_layer(channels_in=filters_out_count, filters_out=filters_out, should_pool=False))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model


def train(hyper_params):
    (train_images, train_labels), (test_images, test_labels) = _get_data()

    model = _build_cnn_model_graph(hyper_params)
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=10,
                        validation_data=(test_images, test_labels))

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(test_acc)

    return history


def plot_history(history):
    plt.plot(history.history['acc'], label='train_accuracy')
    plt.plot(history.history['val_acc'], label = 'validation_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')


if __name__ == '__main__':
    hyper_params = {
        "learning_rate": 0.003,
        "steps": 10000,
        "batch_size": 50,
        "data_set": "mnist",
        "count_2x2_square": 4,
        "count_4x4_circle": 8,
        "count_3x3_square": 12,
        "count_3x3_circle": 8,
        "count_6x6_horizontal_line": 4,
        "count_6x6_vertical_line": 4,
        # "count_2x2_square": 0,
        # "count_4x4_circle": 0,
        # "count_3x3_square": 25,
        # "count_3x3_circle": 0,
        "num_classes": 10,
        "total_pixel_count": 784,
        "fully_connected_1": 0,
        "fully_connected_2": 0,
    }
    history = train(hyper_params)
    plot_history(history)
