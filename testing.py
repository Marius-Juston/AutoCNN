import json
import multiprocessing
import multiprocessing as mp
import os

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from auto_cnn.cnn_structure import CNN, SkipLayer, DefaultOutput
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


def f(x):
    cnn, d = x
    cnn: CNN
    data = d['data']

    cnn.generate()
    cnn.train(data)
    a, b = cnn.evaluate(data)

    d[cnn.hash] = b

    d = dict(d)
    d.pop('data')

    with open("test.json", 'w') as f:
        json.dump(d, f)


def parallel_processing():
    N = 5000
    data = {'x_train': np.random.random((N, 1, 2, 2)), 'y_train': np.random.randint(0, 10, N),
            'x_test': np.random.random((N, 1, 2, 2)), 'y_test': np.random.randint(0, 10, N)}

    number_of_workers = mp.cpu_count()

    a = [CNN((1, 2, 2), DefaultOutput(10), [SkipLayer(512, 215)], load_if_exist=False),
         CNN((1, 2, 2), DefaultOutput(10), [SkipLayer(125, 23)], load_if_exist=False),
         CNN((1, 2, 2), DefaultOutput(10), [SkipLayer(512, 512)], load_if_exist=False)]

    manager = multiprocessing.Manager()
    d = manager.dict()
    d['data'] = data

    cnns = []
    for hash in set(cnn.hash for cnn in a):
        for cnn in a:
            if cnn.hash == hash:
                cnns.append((cnn, d))
                break

    print(number_of_workers)

    with mp.Pool(number_of_workers) as pool:
        pool.map_async(f, cnns)
        pool.close()
        pool.join()

    print(d)


def multiprocess_evaluation():
    N = 5000
    data = {'x_train': np.random.random((N, 1, 2, 2)), 'y_train': np.random.randint(0, 10, N),
            'x_test': np.random.random((N, 1, 2, 2)), 'y_test': np.random.randint(0, 10, N)}

    a = AutoCNN(5, 1, data)
    a.initialize()
    a.evaluate_fitness(a.population)


if __name__ == '__main__':
    mnist_test()
