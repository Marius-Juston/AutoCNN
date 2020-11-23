import os
from abc import abstractmethod, ABC
from typing import Iterable, Callable, Union, Sequence, Dict, Any, Tuple, List

import tensorflow as tf
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2


class Layer(ABC):
    @abstractmethod
    def tensor_rep(self, inputs: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
        """
        Returns the keras Layer representation of the object

        :param inputs: The previous layer to be passed in as input for the next
        :return: the keras Layer representation of the object
        """
        pass


class SkipLayer(Layer):
    GROUP_NUMBER = 1

    def __init__(self, feature_size1: int,
                 feature_size2: int,
                 kernel: Tuple[int, int] = (3, 3),
                 stride: Tuple[int, int] = (1, 1),
                 convolution: str = 'same'):
        """
        Initializes the parameters for the Skip Layer
        The Skip layer comprises of:

        1. A convolution:
            * filter size: feature_size1
            * kernel size: kernel
            * stride size: stride
        2. A Batch Normalization
        3. A ReLU activation
        4. A convolution:
            * filter size: feature_size2
            * kernel size: kernel
            * stride size: stride
        5. A Batch Normalization
        6. Inputs + previous output (the batch norm)
        7. A ReLU activation

        :param feature_size1: the filter size of the first convolution, should be a power of 2
        :param feature_size2: the filter size of the second convolution, should be a power of 2
        :param kernel: the kernel size for all the convolutions, default: (3, 3)
        :param stride: the stride size for all convolutions, default: (1, 1)
        :param convolution: the padding type for all convolution, should be either "valid" or "same"
        """
        self.convolution = convolution
        self.stride = stride
        self.kernel = kernel
        self.feature_size2 = feature_size2
        self.feature_size1 = feature_size1

    def tensor_rep(self, inputs: tf.keras.layers.Layer) -> tf.keras.layers.Activation:
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

    def __repr__(self) -> str:
        return f'{self.feature_size1}-{self.feature_size2}'


class PoolingLayer(Layer):
    pooling_choices = {
        'max': tf.keras.layers.MaxPool2D,
        'mean': tf.keras.layers.AveragePooling2D
    }

    def __init__(self, pooling_type: str, kernel: Tuple[int, int] = (2, 2), stride: Tuple[int, int] = (2, 2)):
        """
        A Pooling layer, this is either a MaxPooling or a AveragePooling layer

        :param pooling_type: either "max" or "mean", this determines the type of pooling layer
        :param kernel: the kernel size for the pooling layer, default: (2, 2)
        :param stride: the stride size for the pooling layer, default: (2, 2)
        """

        self.stride = stride
        self.kernel = kernel
        self.pooling_type = pooling_type

    def tensor_rep(self, inputs: tf.keras.layers.Layer) -> Union[
        tf.keras.layers.MaxPool2D, tf.keras.layers.AveragePooling2D]:
        return PoolingLayer.pooling_choices[self.pooling_type](pool_size=self.kernel, strides=self.stride)(inputs)

    def __repr__(self) -> str:
        return self.pooling_type


class CNN:
    def __init__(self, input_shape: Sequence[int],
                 output_function: Callable[[tf.keras.layers.Layer], tf.keras.layers.Layer],
                 layers: Sequence[Layer],
                 optimizer: OptimizerV2 = None,
                 loss: Union[str, tf.keras.losses.Loss] = 'sparse_categorical_crossentropy',
                 metrics: Iterable[str] = ('accuracy',),
                 load_if_exist: bool = True,
                 extra_callbacks: Iterable[tf.keras.callbacks.Callback] = None,
                 logs_dir: str = './logs/train_data',
                 checkpoint_dir: str = './checkpoints') -> None:
        """
        Initializes the CNN.

        Example for an output layer function, that works with one hot encoded outputs:

        >>> def output_function(inputs):
        ...    out = tf.keras.layers.Flatten()(inputs) # flattens the input since it is going to be 3D tensor
        ...
        ...    return tf.keras.layers.Dense(10, activation='softmax')(out) # this is the final output layer

        >>> CNN((28, 28, 1), output_function, []) # passes the function in without calling it

        :param input_shape: the input shape of the CNN input must be at least a size of 2
        :param output_function: the output function to attach at the end of the layers, this will define what is outputted by the CNN
        :param layers: the layer list to define the CNN
        :param optimizer: the type of optimizer to use when training the CNN
        :param loss: the loss function to use when training the CNN
        :param metrics: the metric to use to quantify how good the CNN is
        :param load_if_exist: if the model already in checkpoint_dir use those weights
        :param extra_callbacks: any other other callbacks to use when training the mode, this could be a learning rate scheduler
        :param logs_dir: the directory where to store the Tensorboard logs
        :param checkpoint_dir: the directory where to story the model checkpoints
        """
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

    def generate(self) -> tf.keras.Model:
        """
        Generate the tf.keras.Model of the CNN based on the layers list, the loss function, the optimizer and the metrics

        :return: the compiled tf.keras.Model
        """

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

    def evaluate(self, data: Dict[str, Any], batch_size: int = 64) -> Tuple[float, float]:
        """
        Evaluates the model, this calculates the accuracy of the model on the test data

        :param data: the data to test on, uses the 'x_test' and the 'y_test' values of data to test on
        :param batch_size: the batch size for the testing
        :return: the loss and the accuracy of the model on the test data
        """

        return self.model.evaluate(data['x_test'], data['y_test'], batch_size=batch_size)

    def train(self, data: Dict[str, Any], batch_size: int = 64, epochs: int = 1) -> None:
        """
        Trains the defined model, cnn.generate() must be called before this function can run.

        The model will split the training data with 20% going on validation, the model will save the one with the
        best validation score after each epoch.

        If the model already exists in the checkpoint_dir defined then it will just load the weights saved instead.

        When training the model uses the TensorBoard and the ModelCheckpoint callbacks to log and save checkpoints of
        the model automatically. You are also able to add any other callbacks through the extra_callback parameters in the CNN __init__

        :param data: the data to train the network on, this is a dict with parameters 'x_train' and 'y_train' which contain the data that will be used in the training period of the model.
        :param batch_size: the batch size to train the network on
        :param epochs: the number of epochs to train the network
        """

        if self.load_if_exist and os.path.exists(f'{self.checkpoint_dir}/model_{self.hash}/'):
            self.model.load_weights(self.checkpoint_filepath)
        else:
            if self.model is not None:
                self.model.fit(data['x_train'], data['y_train'], batch_size=batch_size, epochs=epochs,
                               validation_split=.2,
                               callbacks=self.callbacks)

    def generate_hash(self) -> str:
        """
        Generates the hash of the CNN, this is based on the layers that it contains:

        A SkipLayer is represented as 'feature_size1-feature_size2'

        A PoolingLayer is represented as 'pooling_type'

        Example:
        '32-32-mean-max-256-32'

        :return: the hash of the CNN based on its layer structure
        """

        return '-'.join(map(str, self.layers))

    def __repr__(self) -> str:
        return self.hash


def get_layer_from_string(layer_definition: str) -> List[Layer]:
    """
    Generate the layers list from the string hash, so that it can be passed into a CNN.

    Example:

    '128-64-mean-max-32-32'

    Would be converted to:

    [SkipLayer(128, 64), PoolingLayer('mean'), PoolingLayer('max'), SkipLayer(32, 32)]

    A SkipLayer is represented as 'feature_size1-feature_size2'

    A PoolingLayer is represented as 'pooling_type'

    :param layer_definition: the string representation of the layers
    :return: the list of the converted layers
    """

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
