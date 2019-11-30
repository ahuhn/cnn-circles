from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from comet_ml import Experiment

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

def get_data():
    mnist = input_data.read_data_sets("/tmp/tensorflow/mnist/input_data/", one_hot=True)
    return mnist


def apply_mask(original, filter_shape):
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
    if filter_shape == "square":
        return original
    mask = np.ones(original.shape[1:])
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
    return original * mask


def _get_variable_with_initializer(name, shape, stddev=None, mean=0.0):
    dtype = tf.float32
    if stddev is None:
        initializer = tf.constant_initializer(mean, dtype=dtype)
    else:
        initializer = tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)
    return tf.compat.v1.get_variable(name, shape, initializer=initializer, dtype=dtype)


def _variable_with_weight_decay(name, shape, stddev, wd):
    var = _get_variable_with_initializer(name, shape, stddev=stddev)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name="weight_loss")
        tf.add_to_collection("losses", weight_decay)
    return var


def adjustable_conv2d(conv_in, filter_size, channels_in, channels_out, filter_shape):
    weights = _variable_with_weight_decay(
        "weights",
        shape=[filter_size, filter_size, channels_in, channels_out],
        stddev=5e-2,
        wd=None,
    )
    filtered_weights = apply_mask(weights, filter_shape)
    conv_out = tf.nn.conv2d(conv_in, filtered_weights, [1, 1, 1, 1], padding="SAME")
    return conv_out


def add_conv_layer(conv_layer, conv_in, filter_in_count, filters_out, should_pool):
    conv_out_list = []

    # Each loop takes in conv_in and adds new channels to conv_out. Each needs to be the same size.
    for i, filter_out in enumerate(filters_out):
        filter_out_count = filter_out["filter_count"]
        filter_size = filter_out["filter_size"]
        filter_shape = filter_out["filter_shape"]

        conv_id = f"{i}_{conv_layer}"
        with tf.variable_scope(f"conv_layer_{conv_id}") as scope:
            with tf.variable_scope(f"conv_{conv_id}") as scope:
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
        filters_out.append({"filter_shape": "square", "filter_size": 2, "filter_count": hyper_params["count_2x2_square"]})
    if "count_4x4_circle" in hyper_params and hyper_params["count_4x4_circle"]:
        filters_out.append({"filter_shape": "circle", "filter_size": 4, "filter_count": hyper_params["count_4x4_circle"]})
    if "count_3x3_square" in hyper_params and hyper_params["count_3x3_square"]:
        filters_out.append({"filter_shape": "square", "filter_size": 3, "filter_count": hyper_params["count_3x3_square"]})
    if "count_3x3_circle" in hyper_params and hyper_params["count_3x3_circle"]:
        filters_out.append({"filter_shape": "circle", "filter_size": 3, "filter_count": hyper_params["count_3x3_circle"]})
    return filters_out

def build_cnn_model_graph(hyper_params):
    # Create the model
    x = tf.compat.v1.placeholder(tf.float32, [None, hyper_params["total_pixel_count"]])
    conv_layer_count = hyper_params["conv_layer_count"]
    filters_out = get_filters_out_from_hyper_params(hyper_params)
    conv_out = tf.reshape(x, [-1, 28, 28, 1])
    for conv_layer in range(conv_layer_count):
        filter_in_count = conv_out.shape[-1]
        should_pool = (conv_layer >= (conv_layer_count - 2))
        conv_out = add_conv_layer(
            conv_layer, conv_out, filter_in_count, filters_out, should_pool
        )

    local3_input = conv_out
    with tf.variable_scope("local3") as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        flat_size = local3_input.shape[1] * local3_input.shape[2] * local3_input.shape[3]
        reshaped = tf.reshape(local3_input, [-1, flat_size])
        if hyper_params["fully_connected_1"]:
            dim = reshaped.get_shape()[1].value
            weights = _variable_with_weight_decay(
                "weights", shape=[dim, hyper_params["fully_connected_1"]], stddev=0.04, wd=0.004
            )
            biases = _get_variable_with_initializer("biases", [hyper_params["fully_connected_1"]], mean=0.1)
            local3 = tf.nn.relu(tf.matmul(reshaped, weights) + biases, name=scope.name)
        else:
            local3 = reshaped
    print(local3.shape)

    # local4
    with tf.variable_scope("local4") as scope:
        if hyper_params["fully_connected_2"]:
            dim = local3.get_shape()[1].value
            weights = _variable_with_weight_decay(
                "weights", shape=[dim, hyper_params["fully_connected_2"]], stddev=0.04, wd=0.004
            )
            biases = _get_variable_with_initializer("biases", [hyper_params["fully_connected_2"]], mean=0.1)
            local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
        else:
            local4 = local3
    print(local4.shape)

    # linear layer(WX + b),
    # We don't apply softmax here because
    # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # and performs the softmax internally for efficiency.
    with tf.variable_scope("softmax_linear") as scope:
        dim = local4.get_shape()[1].value
        weights = _variable_with_weight_decay(
            "weights", [dim, hyper_params["num_classes"]], stddev=1 / float(dim), wd=None
        )
        biases = _get_variable_with_initializer("biases", [hyper_params["num_classes"]])
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)

    y = softmax_linear

    # Define loss and optimizer
    y_ = tf.compat.v1.placeholder(tf.float32, [None, hyper_params["num_classes"]])

    cross_entropy = tf.compat.v1.reduce_mean(
        tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))
    train_step = tf.compat.v1.train.GradientDescentOptimizer(hyper_params['learning_rate']).minimize(cross_entropy)

    correct_prediction = tf.compat.v1.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    correct_prediction = tf.compat.v1.cast(correct_prediction, tf.float32)
    accuracy = tf.compat.v1.reduce_mean(correct_prediction)

    return train_step, cross_entropy, accuracy, x, y, y_


# def build_basic_model_graph(hyper_params):
#     # Create the model
#     x = tf.compat.v1.placeholder(tf.float32, [None, 784])
#     W = tf.compat.v1.Variable(tf.zeros([784, 10]))
#     b = tf.compat.v1.Variable(tf.zeros([10]))
#     y = tf.compat.v1.matmul(x, W) + b

#     # Define loss and optimizer
#     y_ = tf.compat.v1.placeholder(tf.float32, [None, 10])

#     cross_entropy = tf.compat.v1.reduce_mean(
#         tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))
#     train_step = tf.compat.v1.train.GradientDescentOptimizer(hyper_params['learning_rate']).minimize(cross_entropy)

#     correct_prediction = tf.compat.v1.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
#     correct_prediction = tf.compat.v1.cast(correct_prediction, tf.float32)
#     accuracy = tf.compat.v1.reduce_mean(correct_prediction)

#     return train_step, cross_entropy, accuracy, x, y, y_

def train(hyper_params):
    mnist = get_data()

    # Get graph definition, tensors and ops
    train_step, cross_entropy, accuracy, x, y, y_ = build_cnn_model_graph(hyper_params)

    experiment = Experiment(api_key="DqTmFnwT1ksBOGat3MV8699kB", project_name="tf_test_cnn", workspace="ahuhn")
    experiment.log_parameters(hyper_params)
    experiment.log_dataset_hash(mnist)

    with tf.Session() as sess:
        with experiment.train():
            sess.run(tf.compat.v1.global_variables_initializer())
            experiment.set_model_graph(sess.graph)

            for i in range(hyper_params["steps"]):
                batch = mnist.train.next_batch(hyper_params["batch_size"])
                experiment.set_step(i)
                # Compute train accuracy every 10 steps
                if i % 100 == 0:
                    train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1]})
                    print('step %d, training accuracy %g' % (i, train_accuracy))
                    experiment.log_metric("accuracy",train_accuracy,step=i)

                # Update weights (back propagation)
                _, loss_val = sess.run([train_step, cross_entropy],
                                       feed_dict={x: batch[0], y_: batch[1]})

                experiment.log_metric("loss",loss_val,step=i)

        ### Finished Training ###

        with experiment.test():
            # Compute test accuracy
            acc = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})
            experiment.log_metric("accuracy",acc)
            print('test accuracy %g' % acc)

if __name__ == '__main__':
    hyper_params = {
        "learning_rate": 0.003,
        "steps": 10000,
        "batch_size": 50,
        "data_set": "mnist",
        # "count_2x2_square": 4,
        # "count_4x4_circle": 8,
        # "count_3x3_square": 12,
        # "count_3x3_circle": 8,
        "count_2x2_square": 0,
        "count_4x4_circle": 0,
        "count_3x3_square": 25,
        "count_3x3_circle": 0,
        "conv_layer_count": 3,
        "num_classes": 10,
        "total_pixel_count": 784,
        "fully_connected_1": 0,
        "fully_connected_2": 0,
    }
    train(hyper_params)