import json
import os
import random
from typing import Dict, Callable, Iterable, Union, Tuple, Sequence, List

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2

tf.get_logger().setLevel('INFO')

from auto_cnn.cnn_structure import SkipLayer, PoolingLayer, CNN, Layer

random.seed(42)


class AutoCNN:
    def get_input_shape(self) -> Tuple[int]:
        shape = self.dataset['x_train'].shape[1:]

        if len(shape) < 3:
            shape = (*shape, 1)

        return shape

    def get_output_function(self) -> Callable[[tf.keras.layers.Layer], tf.keras.layers.Layer]:
        output_size = np.unique(self.dataset['y_train']).shape[0]

        def output_function(inputs):
            out = tf.keras.layers.Flatten()(inputs)

            return tf.keras.layers.Dense(output_size, activation='softmax')(out)

        return output_function

    def __init__(self, population_size: int,
                 maximal_generation_number: int,
                 dataset: Dict[str, np.ndarray],
                 output_layer: Callable[[tf.keras.layers.Layer], tf.keras.layers.Layer] = None,
                 epoch_number: int = 1,
                 optimizer: OptimizerV2 = tf.keras.optimizers.Adam(),
                 loss: Union[str, tf.keras.losses.Loss] = 'sparse_categorical_crossentropy',
                 metrics: Iterable[str] = ('accuracy',),
                 crossover_probability: float = .9,
                 mutation_probability: float = .2,
                 mutation_operation_distribution: Sequence[float] = None,
                 fitness_cache: str = 'fitness.json',
                 extra_callbacks: Iterable[tf.keras.callbacks.Callback] = None,
                 logs_dir: str = './logs/train_data',
                 checkpoint_dir: str = './checkpoints'
                 ) -> None:

        self.logs_dir = logs_dir
        self.checkpoint_dir = checkpoint_dir
        self.extra_callbacks = extra_callbacks
        self.fitness_cache = fitness_cache

        if self.fitness_cache is not None and os.path.exists(self.fitness_cache):
            with open(self.fitness_cache) as cache:
                self.fitness = json.load(cache)
        else:
            self.fitness = dict()

        if mutation_operation_distribution is None:
            self.mutation_operation_distribution = (.7, .1, .1, .1)
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

        self.population_iteration = 0

        if output_layer is None:
            self.output_layer = self.get_output_function()
        else:
            self.output_layer = output_layer

        self.input_shape = self.get_input_shape()

    def initialize(self) -> None:
        self.population.clear()

        for _ in range(self.population_size):
            depth = random.randint(1, 5)

            layers = []

            for i in range(depth):
                r = random.random()

                if r < .5:
                    layers.append(self.random_skip())
                else:
                    layers.append(self.random_pooling())

            cnn = self.generate_cnn(layers)

            self.population.append(cnn)

    def random_skip(self) -> SkipLayer:
        f1 = 2 ** random.randint(5, 9)
        f2 = 2 ** random.randint(5, 9)
        return SkipLayer(f1, f2)

    def random_pooling(self) -> PoolingLayer:
        q = random.random()

        if q < .5:
            return PoolingLayer('max')
        else:
            return PoolingLayer('mean')

    def evaluate_fitness(self, population: Iterable[CNN]) -> None:
        for cnn in population:
            if cnn.hash not in self.fitness:
                # TODO make this work on multiple GPUs simultaneously
                self.evaluate_individual_fitness(cnn)

            print(cnn, self.fitness[cnn.hash])

    def evaluate_individual_fitness(self, cnn: CNN) -> None:
        try:
            cnn.generate()

            cnn.train(self.dataset, epochs=self.epoch_number)
            loss, accuracy = cnn.evaluate(self.dataset)
        except ValueError as e:
            print(e)
            accuracy = 0

        self.fitness[cnn.hash] = accuracy

        if self.fitness_cache is not None:
            with open(self.fitness_cache, 'w') as json_file:
                json.dump(self.fitness, json_file)

    def select_two_individuals(self, population: Sequence[CNN]) -> CNN:
        cnn1, cnn2 = random.sample(population, 2)

        if self.fitness[cnn1.hash] > self.fitness[cnn2.hash]:
            return cnn1
        else:
            return cnn2

    def split_individual(self, cnn: CNN) -> Tuple[List[Layer], List[Layer]]:
        split_index = random.randint(0, len(cnn.layers))

        return cnn.layers[:split_index], cnn.layers[split_index:]

    def generate_offsprings(self) -> Sequence[CNN]:
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

        offsprings = [self.generate_cnn(layers) for layers in offsprings]

        return offsprings

    def generate_cnn(self, layers: Sequence[Layer]) -> CNN:
        return CNN(self.input_shape, self.output_layer, layers, optimizer=self.optimizer, loss=self.loss,
                   metrics=self.metrics, extra_callbacks=self.extra_callbacks, logs_dir=self.logs_dir,
                   checkpoint_dir=self.checkpoint_dir)

    def environmental_selection(self, offsprings: Sequence[CNN]) -> Iterable[CNN]:
        whole_population = list(self.population)
        whole_population.extend(offsprings)

        new_population = []

        while len(new_population) < len(self.population):
            p = self.select_two_individuals(whole_population)

            new_population.append(p)

        best_cnn = max(whole_population, key=lambda x: self.fitness[x.hash])

        print("Best CNN:", best_cnn, "Score:", self.fitness[best_cnn.hash])

        if best_cnn not in new_population:
            worst_cnn = min(new_population, key=lambda x: self.fitness[x.hash])
            print("Worst CNN:", worst_cnn, "Score:", self.fitness[worst_cnn.hash])
            new_population.remove(worst_cnn)
            new_population.append(best_cnn)

        return new_population

    def run(self) -> CNN:
        print("Initializing Population")
        self.initialize()
        print("Population Initialization Done:", self.population)

        for i in range(self.maximal_generation_number):
            print("Generation", i)

            print("Evaluating Population fitness")
            self.evaluate_fitness(self.population)
            print("Evaluating Population fitness Done:", self.fitness)

            print("Generating Offsprings")
            offsprings = self.generate_offsprings()
            print("Generating Offsprings Done:", offsprings)

            print("Evaluating Offsprings")
            self.evaluate_fitness(offsprings)
            print("Evaluating Offsprings Done:", self.fitness)

            print("Selecting new environment")
            new_population = self.environmental_selection(offsprings)
            print("Selecting new environment Done:", new_population)

            self.population = new_population

        best_cnn = sorted(self.population, key=lambda x: self.fitness[x.hash])[-1]
        print("Best CNN:", best_cnn, "Score:", self.fitness[best_cnn.hash])
        return best_cnn


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
    cifar10_test()
