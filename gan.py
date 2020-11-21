import json
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
                 mutation_probability: float = .2, mutation_operation_distribution: Iterable[float] = None,
                 fitness_cache='fitness.json'):

        self.fitness_cache = fitness_cache

        if self.fitness_cache is not None:
            with open(self.fitness_cache) as cache:
                self.fitness = json.load(cache)
        else:
            self.fitness = dict()

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
                    layers.append(self.random_skip())
                else:
                    layers.append(self.random_pooling())

            cnn = CNN(self.input_shape, self.output_layer, layers, optimizer=self.optimizer, loss=self.loss,
                      metrics=self.metrics)

            self.population.append(cnn)

    def random_skip(self):
        f1 = 2 ** random.randint(5, 9)
        f2 = 2 ** random.randint(5, 9)
        return SkipLayer(f1, f2)

    def random_pooling(self):
        q = random.random()

        if q < .5:
            return PoolingLayer('max')
        else:
            return PoolingLayer('mean')

    def evaluate_fitness(self, population):
        for cnn in population:
            if cnn.hash not in self.fitness:
                # TODO make this work on multiple GPUs simultaneously
                self.evaluate_individual_fitness(cnn)

    def evaluate_individual_fitness(self, cnn: CNN):
        cnn.generate()
        cnn.train(data, epochs=self.epoch_number)
        loss, accuracy = cnn.evaluate(data)

        self.fitness[cnn.hash] = accuracy

        if self.fitness_cache is not None:
            with open(self.fitness_cache, 'w') as json_file:
                json.dump(self.fitness, json_file)

    def select_two_individuals(self, population):
        cnn1, cnn2 = random.sample(population, 2)

        if self.fitness[cnn1.hash] > self.fitness[cnn2.hash]:
            return cnn1
        else:
            return cnn2

    def split_individual(self, cnn: CNN):
        split_index = random.randint(0, len(cnn.layers))

        return cnn.layers[:split_index], cnn.layers[split_index:]

    def generate_offsprings(self):
        offsprings = []

        while len(offsprings) < len(self.population):
            p1 = self.select_two_individuals(self.population)
            p2 = self.select_two_individuals(self.population)

            while p1.hash == p2.hash:
                p2 = self.select_two_individuals(self.population)

            r = random.random()

            if r < self.crossover_probability:
                p1_1, p1_2 = self.split_individual(p1)
                p2_1, p2_2 = self.split_individual(p2)

                p1_1.extend(p2_2)
                p2_1.extend(p1_2)

                offsprings.append(p1_1)
                offsprings.append(p2_1)
            else:
                offsprings.append(p1.layers)
                offsprings.append(p2.layers)

        print(offsprings)

        choices = ['add_skip', 'add_pooling', 'remove', 'change']

        for cnn in offsprings:
            cnn: list

            r = random.random()

            if r < self.mutation_probability:
                if len(cnn) == 0:
                    i = 0
                    operation = random.choices(choices[:2], weights=self.mutation_operation_distribution[:2])[0]
                else:
                    i = random.randint(0, len(cnn) - 1)
                    operation = random.choices(choices, weights=self.mutation_operation_distribution)[0]

                if operation == 'add_skip':
                    cnn.insert(i, self.random_skip())
                elif operation == 'add_pooling':
                    cnn.insert(i, self.random_pooling())
                elif operation == 'remove':
                    cnn.pop(i)
                else:
                    if isinstance(cnn[i], SkipLayer):
                        cnn[i] = self.random_skip()
                    else:
                        cnn[i] = self.random_pooling()

        offsprings = [CNN(self.input_shape, self.output_layer, layers, optimizer=self.optimizer, loss=self.loss,
                          metrics=self.metrics) for layers in offsprings]

        return offsprings


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    data = {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}

    a = AutoCNN(5, 1, data)
    a.evaluate_fitness()
