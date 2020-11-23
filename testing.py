import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

tf.get_logger().setLevel('INFO')

from auto_cnn.gan import AutoCNN

import random

random.seed(42)
tf.random.set_seed(42)


def mnist_test():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    values = x_train.shape[0] // 2

    data = {'x_train': x_train[:values], 'y_train': y_train[:values], 'x_test': x_test, 'y_test': y_test}

    a = AutoCNN(5, 1, data)
    a.run()


def cifar10_test():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    values = x_train.shape[0]

    data = {'x_train': x_train[:values], 'y_train': y_train[:values], 'x_test': x_test, 'y_test': y_test}

    a = AutoCNN(20, 10, data, epoch_number=10)
    a.run()


if __name__ == '__main__':
    mnist_test()
