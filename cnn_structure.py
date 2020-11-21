from abc import abstractmethod, ABC

import tensorflow as tf


class Layer(ABC):
    @abstractmethod
    def tensor_rep(self, inputs):
        pass


class SkipLayer(Layer):
    def __init__(self, feature_size1, feature_size2, kernel=(3, 3), stride=(1, 1), convolution='same'):
        self.convolution = convolution
        self.stride = stride
        self.kernel = kernel
        self.feature_size2 = feature_size2
        self.feature_size1 = feature_size1

    def tensor_rep(self, inputs):
        skip_layer = tf.keras.layers.Conv2D(self.feature_size1, self.kernel, self.stride, self.convolution,
                                            activation='relu')(inputs)
        skip_layer = tf.keras.layers.Conv2D(self.feature_size2, self.kernel, self.stride, self.convolution)(skip_layer)

        # Makes sure that the dimensionality at the skip layers are the same
        inputs = tf.keras.layers.Conv2D(self.feature_size2, (1, 1), self.stride)(inputs)

        outputs = tf.keras.layers.add([inputs, skip_layer])
        return tf.keras.layers.Activation('relu')(outputs)

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
    def __init__(self, input_shape, output_function, layers=None, optimizer=tf.keras.optimizers.Adam(),
                 loss='sparse_categorical_crossentropy', metrics=('accuracy',)):
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.output_function = output_function
        self.input_shape = input_shape
        if layers is None:
            self.layers = []
        else:
            self.layers = layers

        self.model: tf.keras.Model = None

    def generate(self):
        print(self.layers)
        if self.model is None:
            inputs = tf.keras.Input(shape=self.input_shape)

            outputs = inputs

            for i, layer in enumerate(self.layers):
                outputs = layer.tensor_rep(outputs)

            # TODO Change this to a more variable output!!!
            outputs = self.output_function(outputs)

            self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
            self.model.summary()

        return self.model

    def train(self, data, batch_size=64, epochs=1):
        self.model.compile(self.optimizer, loss=self.loss, metrics=self.metrics)

        if self.model is not None:
            self.model.fit(data['x_train'], data['y_train'], batch_size=batch_size, epochs=epochs, validation_split=.2)

    def __repr__(self):
        return '-'.join(map(str, self.layers))

    def append(self, layer):
        self.layers.append(layer)


if __name__ == '__main__':
    def output_function(inputs):
        out = tf.keras.layers.Flatten()(inputs)

        return tf.keras.layers.Dense(10, activation='softmax')(out)


    cnn = CNN((28, 28, 1), output_function,
              layers=[SkipLayer(32, 64), PoolingLayer('max'), SkipLayer(128, 256), PoolingLayer('mean'),
                      SkipLayer(512, 256), SkipLayer(256, 512)])

    cnn.generate()

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    print(x_train.shape)

    data = {'x': x_train, 'y': y_train}

    cnn.train(data)