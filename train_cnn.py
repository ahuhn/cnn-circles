from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from comet_ml import Experiment

import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

CIFAR10_CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


def get_data():
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0
    return (train_images, train_labels), (test_images, test_labels)


def get_mask(filter_size, channels_in, filter_shape):
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
    mask = np.ones([filter_size[0], filter_size[1], channels_in])
    if filter_shape == "square":
        pass
    elif filter_shape == "circle":
        mask[0][0] = 0
        mask[0][-1] = 0
        mask[-1][0] = 0
        mask[-1][-1] = 0
        if original.shape[0] > 5:
            mask[0][1] = 0
            mask[1][0] = 0
            mask[0][-2] = 0
            mask[1][-1] = 0
            mask[-1][1] = 0
            mask[-2][0] = 0
            mask[-1][-2] = 0
            mask[-2][-1] = 0
    return mask


class MaskLayer(tf.keras.layers.Layer):
    def __init__(self, filter_size, filter_shape, **kwargs):
        self.filter_size = filter_size
        self.channels_in = channels_in
        self.filter_shape = filter_shape
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        channels_in = input_shape[-1]
        self.kernel = get_mask(self.filter_size, channels_in, self.filter_shape)
        super(MyLayer, self).build(input_shape)

    def call(self, x):
        return tf.keras.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return input_shape


def add_conv_layer(model, conv_in, filter_in_count, filters_out, should_pool):
    conv_out_list = []

    # Each loop takes in conv_in and adds new channels to conv_out. Each needs to be the same size.
    for i, filter_out in enumerate(filters_out):
        filter_out_count = filter_out["filter_count"]
        filter_size = filter_out["filter_size"]
        filter_shape = filter_out["filter_shape"]

        conv = layers.Conv2D(filter_out["filter_count"], filter_out["filter_size"], activation='relu', input_shape=(32, 32, 3))
        filtered_conv = MaskLayer()
        pooled_conv = layers.MaxPooling2D((2, 2))(filtered_conv)

        conv = adjustable_conv2d(conv_in, filter_size, filter_in_count, filter_out_count, filter_shape)
        biases = _get_variable_with_initializer("biases", [filter_out_count])
        pre_activation = tf.nn.bias_add(conv, biases)
        conv_out = tf.nn.relu(pre_activation, name=scope.name)

        if should_pool:
            conv_out = tf.nn.max_pool(
                conv_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name=f"pool_{conv_id}"
            )
        # TODO: decide when to apply the norm, relative to the outer for-loop
        conv_out = tf.nn.lrn(conv_out, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=f"norm_{conv_id}")
        conv_out_list.append(conv_out)

    print("Shapes")
    for conv_out in conv_out_list:
        print(conv_out.shape)
#     return conv_out_list
    return tf.compat.v1.concat(conv_out_list, -1)

def get_filters_out_from_hyper_params(hyper_params):
    filters_out = []
    if "count_2x2_square" in hyper_params and hyper_params["count_2x2_square"]:
        filters_out.append({"filter_shape": "square", "filter_size": (2, 2), "filter_count": hyper_params["count_2x2_square"]})
    if "count_4x4_circle" in hyper_params and hyper_params["count_4x4_circle"]:
        filters_out.append({"filter_shape": "circle", "filter_size": (4, 4), "filter_count": hyper_params["count_4x4_circle"]})
    if "count_3x3_square" in hyper_params and hyper_params["count_3x3_square"]:
        filters_out.append({"filter_shape": "square", "filter_size": (3, 3), "filter_count": hyper_params["count_3x3_square"]})
    if "count_3x3_circle" in hyper_params and hyper_params["count_3x3_circle"]:
        filters_out.append({"filter_shape": "circle", "filter_size": (3, 3), "filter_count": hyper_params["count_3x3_circle"]})
    if "count_6x6_horizontal_line" in hyper_params and hyper_params["count_6x6_horizontal_line"]:
        filters_out.append({"filter_shape": "horizontal_line", "filter_size": (6, 2), "filter_count": hyper_params["count_6x6_circle"]})
    if "count_6x6_vertical_line" in hyper_params and hyper_params["count_6x6_vertical_line"]:
        filters_out.append({"filter_shape": "vertical_line", "filter_size": (2, 6), "filter_count": hyper_params["count_6x6_circle"]})
    return filters_out

def build_cnn_model_graph(hyper_params):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model


def train(hyper_params):
    (train_images, train_labels), (test_images, test_labels) = get_data()

    model = build_cnn_model_graph(hyper_params)
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=10,
                        validation_data=(test_images, test_labels))

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(test_acc)

    return history

    # # Get graph definition, tensors and ops
    # train_step, cross_entropy, accuracy, x, y, y_ = build_cnn_model_graph(hyper_params)

    # experiment = Experiment(api_key="DqTmFnwT1ksBOGat3MV8699kB", project_name="tf_test_cnn", workspace="ahuhn")
    # experiment.log_parameters(hyper_params)
    # experiment.log_dataset_hash(training_data)

    # with tf.compat.v1.Session() as sess:
    #     with experiment.train():
    #         sess.run(tf.compat.v1.global_variables_initializer())
    #         experiment.set_model_graph(sess.graph)

    #         for i in range(hyper_params["steps"]):
    #             batch = training_data.take(1)
    #             experiment.set_step(i)
    #             # Compute train accuracy every 10 steps
    #             if i % 100 == 0:
    #                 train_accuracy = accuracy.eval(feed_dict={x: batch["image"], y_: batch["label"]})
    #                 print('step %d, training accuracy %g' % (i, train_accuracy))
    #                 experiment.log_metric("accuracy",train_accuracy,step=i)

    #             # Update weights (back propagation)
    #             _, loss_val = sess.run([train_step, cross_entropy],
    #                                    feed_dict={x: batch[0], y_: batch[1]})

    #             experiment.log_metric("loss",loss_val,step=i)

    #     ### Finished Training ###

    #     with experiment.test():
    #         # Compute test accuracy
    #         test_batch = test_data.take(1)
    #         acc = accuracy.eval(feed_dict={x: test_batch["images"], y_: test_batch["labels"]})
    #         experiment.log_metric("accuracy",acc)
    #         print('test accuracy %g' % acc)

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
        "conv_layer_count": 2,
        "num_classes": 10,
        "total_pixel_count": 784,
        "fully_connected_1": 0,
        "fully_connected_2": 0,
    }
    history = train(hyper_params)
    plot_history(history)
