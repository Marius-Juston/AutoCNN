import random
from typing import Dict, Callable, Iterable

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

    def get_output_function(self):
        output_size = len(set(self.dataset['y_train']))

        def output_function(inputs):
            out = tf.keras.layers.Flatten()(inputs)

            return tf.keras.layers.Dense(output_size, activation='softmax')(out)

        return output_function

    def __init__(self, population_size: int, maximal_generation_number: int, dataset: Dict[str, np.ndarray],
                 output_layer: Callable[[tf.keras.layers.Layer], tf.keras.layers.Layer] = None, epoch_number: int = 1,
                 optimizer=tf.keras.optimizers.SGD(.1, .9),
                 loss='sparse_categorical_crossentropy', metrics=('accuracy',), crossover_probability: float = .9,
                 mutation_probability: float = .2, mutation_operation_distribution: Iterable[float] = None):

        l = 4

        if mutation_operation_distribution is None:
            self.mutation_operation_distribution = [1 / l for _ in range(l)]
        else:
            self.mutation_operation_distribution = mutation_operation_distribution

        self.mutation_probability = mutation_probability
        self.crossover_probability = crossover_probability
        self.epoch_number = epoch_number
        self.metrics = metrics
        self.loss = loss
        self.optimizer = optimizer
        self.dataset = dataset
        self.maximal_generation_number = maximal_generation_number
        self.population_size = population_size
        self.population = []
        self.fitness = dict()

        self.population_iteration = 0

        if output_layer is None:
            self.output_layer = self.get_output_function()
        else:
            self.output_layer = output_layer

        self.input_shape = self.get_input_shape()

        self.initialize()

    def initialize(self):
        self.population.clear()

        for _ in range(self.population_size):
            depth = random.randint(1, 10)

            layers = []

            for i in range(depth):
                r = random.random()

                if r < .5:
                    f1 = 2 ** random.randint(5, 9)
                    f2 = 2 ** random.randint(5, 9)

                    layers.append(SkipLayer(f1, f2))

                else:
                    q = random.random()

                    if q < .5:
                        layers.append(PoolingLayer('max'))
                    else:
                        layers.append(PoolingLayer('mean'))

            cnn = CNN(self.input_shape, self.output_layer, layers, optimizer=self.optimizer, loss=self.loss,
                      metrics=self.metrics)

            self.population.append(cnn)

    def evaluate_fitness(self):
        for cnn in self.population:
            if str(cnn) not in self.fitness:
                # TODO make this work on multiple GPUs simultaneously
                self.evaluate_individual_fitness(cnn)

    def evaluate_individual_fitness(self, cnn: CNN):
        cnn.generate()
        cnn.train(data, epochs=self.epoch_number)
        loss, accuracy = cnn.evaluate(data)

        self.fitness[str(cnn)] = accuracy

    def generate_offsprings(self):
        pass


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    data = {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}

    a = AutoCNN(5, 1, data)
    a.evaluate_fitness()
