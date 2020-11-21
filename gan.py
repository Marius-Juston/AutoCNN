import random
from typing import Dict, Callable

import numpy as np
import tensorflow as tf

from cnn_structure import SkipLayer, PoolingLayer, CNN

random.seed(42)


class AutoCNN:
    def get_input_shape(self):
        shape = self.dataset['x_train'].shape[1:]

        if len(shape) < 3:
            shape = (*shape, 1)

        return shape

    def __init__(self, population_size: int, maximal_generation_number: int, dataset: Dict[str, np.ndarray],
                 output_layer: Callable[[tf.keras.layers.Layer], tf.keras.layers.Layer],
                 optimizer=tf.keras.optimizers.Adam(),
                 loss='sparse_categorical_crossentropy', metrics=('accuracy',)):
        self.metrics = metrics
        self.loss = loss
        self.optimizer = optimizer
        self.output_layer = output_layer
        self.dataset = dataset

        self.maximal_generation_number = maximal_generation_number
        self.population_size = population_size
        self.population = []

        self.input_shape = self.get_input_shape()

        self.initialize()

    def initialize(self):
        self.population.clear()

        for _ in range(self.population_size):
            depth = random.randint(1, 10)
            cnn = CNN(self.input_shape, self.output_layer, optimizer=self.optimizer, loss=self.loss,
                      metrics=self.metrics)

            for i in range(depth):
                r = random.random()

                if r < .5:
                    f1 = 2 ** random.randint(5, 9)
                    f2 = 2 ** random.randint(5, 9)

                    cnn.append(SkipLayer(f1, f2))

                else:
                    q = random.random()

                    if q < .5:
                        cnn.append(PoolingLayer('max'))
                    else:
                        cnn.append(PoolingLayer('mean'))

            self.population.append(cnn)


if __name__ == '__main__':
    def output_function(inputs):
        out = tf.keras.layers.Flatten()(inputs)

        return tf.keras.layers.Dense(10, activation='softmax')(out)


    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    data = {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}

    a = AutoCNN(5, 1, data, output_function)

    print(a.population)
    a.population[0].generate()