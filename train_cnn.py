from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from comet_ml import Experiment

import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models, constraints, Input, Sequential
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


# class FilterShapeConstraint(constraints.Constraint):
#     """
#     Adjusts the shape of the filter by multiplying some weights by 0
#     """

#     def __init__(self, filter_shape_name, filter_dims, channels_in):
#         self.filter_shape_name = filter_shape_name
#         self.filter_dims = filter_dims
#         self.mask = _get_mask(filter_shape_name, filter_dims, channels_in)

#     def __call__(self, w):
#         return w * self.mask

#     def get_config(self):
#         return {"filter_shape_name": self.filter_shape_name, "filter_dims": self.filter_dims, "channels_in": channels_in}

class ConvLayerWithFilters(layers.Layer):
    def __init__(self, filters_out):
        super(ConvLayerWithFilters, self).__init__()
        self.filters_out = filters_out
        print("trainable", self.trainable)

    def build(self, input_shape):
        self.sublayer_list = []
        print("input_shape", input_shape)

        channels_in = input_shape[-1]
        conv = layers.Conv2D(64, (3, 3), padding="same", activation='relu', input_shape=(32, 32, channels_in))
        self.sublayer_list.append(conv)
        # for i, filter_out in enumerate(self.filters_out):
        #     # constraint = FilterShapeConstraint(filter_out["filter_shape_name"], filter_out["filter_dims"], channels_in)
        #     # conv = layers.Conv2D(filter_out["filter_count"], filter_out["filter_dims"], padding="same", activation='relu', kernel_constraint=constraint, input_shape=input_shape)
        #     conv = layers.Conv2D(64, (3, 3), padding="same", activation='relu', input_shape=(32, 32, 3))
        #     pool = layers.MaxPooling2D((2, 2))
        #     normalize = None # layers.BatchNormalization()
        #     self.sublayer_list.append((conv, pool, normalize))

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
        return {"filters_out": filters_out}

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
    if "count_6x6_line1" in hyper_params and hyper_params["count_6x6_line1"]:
        filters_out.append({"filter_shape_name": "line1", "filter_dims": (6, 2), "filter_count": hyper_params["count_6x6_line1"]})
    if "count_6x6_line2" in hyper_params and hyper_params["count_6x6_line2"]:
        filters_out.append({"filter_shape_name": "line2", "filter_dims": (2, 6), "filter_count": hyper_params["count_6x6_line2"]})
    return filters_out

def _build_cnn_model_graph(hyper_params):
    filters_out = _get_filters_out_from_hyper_params(hyper_params)
    model = Sequential()

    # model.add(layers.Conv2D(32, (3, 3), padding="same", activation='relu', input_shape=(32, 32, 3)))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(64, (3, 3), padding="same", activation='relu'))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(64, (3, 3), padding="same", activation='relu'))
    model.add(ConvLayerWithFilters(filters_out=filters_out))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(ConvLayerWithFilters(filters_out=filters_out))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(ConvLayerWithFilters(filters_out=filters_out))
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

    history = model.fit(train_images, train_labels, epochs=1,
                        validation_data=(test_images, test_labels))
    print(model.summary())

    # plot_history(history)

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
    plt.show()


if __name__ == '__main__':
    hyper_params = {
        # "count_2x2_square": 4,
        # "count_4x4_circle": 8,
        # "count_3x3_square": 12,
        # "count_3x3_circle": 8,
        # "count_6x6_line1": 4,
        # "count_6x6_line2": 4,
        # "count_3x3_square": 25,
        "count_3x3_square": 64,
    }
    history = train(hyper_params)
    # plot_history(history)
