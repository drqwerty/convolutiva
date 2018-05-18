# -*- coding: utf-8 -*-

# Sample code to use string producer.

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    o_h = np.zeros(n)
    o_h[x] = 1
    return o_h


num_classes = 3
batch_size = 4


# --------------------------------------------------
#
#       DATA SOURCE
#
# --------------------------------------------------

def dataSource(paths, batch_size):
    min_after_dequeue = 10  # train start delay
    capacity = min_after_dequeue + 3 * batch_size  # max size

    example_batch_list = []
    label_batch_list = []

    # open files
    for i, p in enumerate(paths):
        filename = tf.train.match_filenames_once(p)
        filename_queue = tf.train.string_input_producer(filename, shuffle=False)
        reader = tf.WholeFileReader()
        # image decode
        _, file_image = reader.read(filename_queue)
        image, label = tf.image.decode_jpeg(file_image), [float(i)]  # [one_hot(float(i), num_classes)]
        image = tf.image.resize_image_with_crop_or_pad(image, 80, 140)
        # 24 to 8 bits
        image = tf.image.rgb_to_grayscale(image)
        image = tf.reshape(image, [80, 140, 1])  # 2d to 3d tensor
        image = tf.to_float(image) / 255. - 0.5
        # feed buffer it up capacity
        example_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity,
                                                            min_after_dequeue=min_after_dequeue)
        example_batch_list.append(example_batch)
        label_batch_list.append(label_batch)  # one_hot labels

    # tensor joiner
    example_batch = tf.concat(values=example_batch_list, axis=0)
    label_batch = tf.concat(values=label_batch_list, axis=0)

    return example_batch, label_batch


# --------------------------------------------------
#
#       MODEL
#
# --------------------------------------------------

def myModel(X, reuse=False):
    with tf.variable_scope('ConvNet', reuse=reuse):
        o1 = tf.layers.conv2d(inputs=X, filters=32, kernel_size=3, activation=tf.nn.relu)
        o2 = tf.layers.max_pooling2d(inputs=o1, pool_size=2, strides=2)
        o3 = tf.layers.conv2d(inputs=o2, filters=64, kernel_size=3, activation=tf.nn.relu)
        o4 = tf.layers.max_pooling2d(inputs=o3, pool_size=2, strides=2)

        # exit nums, elements that we'll use
        h = tf.layers.dense(inputs=tf.reshape(o4, [batch_size * num_classes, 18 * 33 * 64]), units=5,
                            activation=tf.nn.relu)
        y = tf.layers.dense(inputs=h, units=num_classes,
                            activation=tf.nn.softmax)  # sigmoid only accepts 1, for that is 'tf.nn.softmax'
    return y


example_batch_train, label_batch_train = dataSource(["dataset/0/train/*.jpg", "dataset/1/train/*.jpg", "dataset/2/train/*.jpg"], batch_size=batch_size)
example_batch_valid, label_batch_valid = dataSource(["dataset/0/valid/*.jpg", "dataset/1/valid/*.jpg", "dataset/2/valid/*.jpg"], batch_size=batch_size)
example_batch_test, label_batch_test = dataSource(["dataset/0/test/*.jpg", "dataset/1/test/*.jpg", "dataset/2/test/*.jpg"], batch_size=batch_size)

# reuse is true for avoid mix data in the same model,
example_batch_train_predicted = myModel(example_batch_train, reuse=False)
example_batch_valid_predicted = myModel(example_batch_valid, reuse=True)
example_batch_test_predicted = myModel(example_batch_test, reuse=True)

# train cost 'square sum'
cost = tf.reduce_sum(tf.square(example_batch_train_predicted - label_batch_train))
cost_valid = tf.reduce_sum(tf.square(example_batch_valid_predicted - label_batch_valid))
cost_test = tf.reduce_sum(tf.square(example_batch_test_predicted - label_batch_test))
# cost = tf.reduce_mean(-tf.reduce_sum(label_batch * tf.log(y), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(
    cost)  # learning_rate; amount inversely proportional to learning

# --------------------------------------------------
#
#       TRAINING
#
# --------------------------------------------------

# Add ops to save and restore all the variables.

saver = tf.train.Saver()

with tf.Session() as sess:
    file_writer = tf.summary.FileWriter('./logs', sess.graph)

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # Start populating the filename queue.
    # thread manager
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    error_train_values = []
    error_validation_values = []

    for _ in range(430):
        sess.run(optimizer)
        error_train = sess.run(cost)
        error_validation = sess.run(cost_valid)
        if _ % 20 == 0:
            # print("Iter:", _, "---------------------------------------------")
            # print(sess.run(label_batch_valid))
            # print(sess.run(example_batch_valid_predicted))
            # print("Error:", sess.run(cost_valid))
            if _ > 1:
                diff = abs(error_validation_values[-2] - error_validation_values[-1])
                if diff < .0001:
                    break
            print("Iter:", _, "Error train:", error_train)
            print("\tError validation:", error_validation)
        error_train_values.append(error_train)
        error_validation_values.append(error_validation)


    save_path = saver.save(sess, "./tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)

    training_plot, = plt.plot(error_train_values)
    validation_plot, = plt.plot(error_validation_values)
    plt.legend(handles=[training_plot, validation_plot],
                labels=['Error', 'Validation'])
    plt.title("Training vs Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.show()

    coord.request_stop()
    coord.join(threads)