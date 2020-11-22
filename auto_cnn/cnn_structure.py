import os
from abc import abstractmethod, ABC

import tensorflow as tf

tf.random.set_seed(42)


class Layer(ABC):
    @abstractmethod
    def tensor_rep(self, inputs):
        pass


class SkipLayer(Layer):
    GROUP_NUMBER = 1

    def __init__(self, feature_size1, feature_size2, kernel=(3, 3), stride=(1, 1), convolution='same'):
        self.convolution = convolution
        self.stride = stride
        self.kernel = kernel
        self.feature_size2 = feature_size2
        self.feature_size1 = feature_size1

    def tensor_rep(self, inputs):
        group_name = f'SkipLayer_{SkipLayer.GROUP_NUMBER}'
        SkipLayer.GROUP_NUMBER += 1

        skip_layer = tf.keras.layers.Conv2D(self.feature_size1, self.kernel, self.stride, self.convolution,
                                            name=f'{group_name}/Conv1')(inputs)

        # FIXME is it the activation first or the batch norm?
        skip_layer = tf.keras.layers.BatchNormalization(name=f'{group_name}/BatchNorm1')(skip_layer)
        skip_layer = tf.keras.layers.Activation('relu', name=f'{group_name}/ReLU1')(skip_layer)

        skip_layer = tf.keras.layers.Conv2D(self.feature_size2, self.kernel, self.stride, self.convolution,
                                            name=f'{group_name}/Conv2')(skip_layer)
        skip_layer = tf.keras.layers.BatchNormalization(name=f'{group_name}/BatchNorm2')(skip_layer)

        # Makes sure that the dimensionality at the skip layers are the same
        inputs = tf.keras.layers.Conv2D(self.feature_size2, (1, 1), self.stride, name=f'{group_name}/Reshape')(inputs)

        outputs = tf.keras.layers.add([inputs, skip_layer], name=f'{group_name}/Add')
        return tf.keras.layers.Activation('relu', name=f'{group_name}/ReLU2')(outputs)

    def __repr__(self):
        return f'{self.feature_size1}-{self.feature_size2}'


class PoolingLayer(Layer):
    pooling_choices = {
        'max': tf.keras.layers.MaxPool2D,
        'mean': tf.keras.layers.AveragePooling2D
    }

    def __init__(self, pooling_type, kernel=(2, 2), stride=(2, 2)):
        self.stride = stride
        self.kernel = kernel
        self.pooling_type = pooling_type

    def tensor_rep(self, inputs):
        return PoolingLayer.pooling_choices[self.pooling_type](pool_size=self.kernel, strides=self.stride)(inputs)

    def __repr__(self):
        return self.pooling_type


class CNN:
    def __init__(self, input_shape, output_function, layers, optimizer=None,
                 loss='sparse_categorical_crossentropy', metrics=('accuracy',), load_if_exist=True,
                 extra_callbacks=None, logs_dir='./logs/train_data', checkpoint_dir='./checkpoints'):
        self.checkpoint_dir = checkpoint_dir
        self.logs_dir = logs_dir
        self.load_if_exist = load_if_exist
        self.loss = loss

        if optimizer is None:
            self.optimizer = tf.keras.optimizers.Adam()
        else:
            self.optimizer = optimizer

        self.metrics = metrics
        self.output_function = output_function
        self.input_shape = input_shape
        if layers is None:
            self.layers = []
        else:
            self.layers = layers

        self.hash = self.generate_hash()

        self.model: tf.keras.Model = None

        # TODO change this so that the checkpoint works no matter when you change layer
        self.checkpoint_filepath = f'{self.checkpoint_dir}/model_{self.hash}/model_{self.hash}'
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_filepath,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f"{self.logs_dir}/model_{self.hash}",
                                                              update_freq='batch', histogram_freq=1)

        self.callbacks = [model_checkpoint_callback, tensorboard_callback]

        if extra_callbacks is not None:
            self.callbacks.extend(extra_callbacks)

    def generate(self):
        print(self.layers)

        if self.model is None:
            tf.keras.backend.clear_session()  # Fixes graph appending
            SkipLayer.GROUP_NUMBER = 1
            inputs = tf.keras.Input(shape=self.input_shape)

            outputs = inputs

            for i, layer in enumerate(self.layers):
                outputs = layer.tensor_rep(outputs)

            outputs = self.output_function(outputs)

            self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
            # self.model.summary()
            self.model.compile(self.optimizer, loss=self.loss, metrics=self.metrics)

            SkipLayer.GROUP_NUMBER = 1
        return self.model

    def evaluate(self, data, batch_size=64):
        return self.model.evaluate(data['x_test'], data['y_test'], batch_size=batch_size)

    def train(self, data, batch_size=64, epochs=1):
        if self.load_if_exist and os.path.exists(f'{self.checkpoint_dir}/model_{self.hash}/'):
            self.model.load_weights(self.checkpoint_filepath)
        else:
            if self.model is not None:
                self.model.fit(data['x_train'], data['y_train'], batch_size=batch_size, epochs=epochs,
                               validation_split=.2,
                               callbacks=self.callbacks)

    def generate_hash(self):
        return '-'.join(map(str, self.layers))

    def __repr__(self):
        return self.hash


def get_layer_from_string(layer_definition):
    layers_str: list = layer_definition.split('-')

    layers = []

    while len(layers_str) > 0:
        if layers_str[0].isdigit():
            f = SkipLayer(int(layers_str[0]), int(layers_str[0 + 1]))
            layers_str.pop(0)
            layers_str.pop(0)
        else:
            f = PoolingLayer(layers_str[0])
            layers_str.pop(0)
        layers.append(f)

    return layers
