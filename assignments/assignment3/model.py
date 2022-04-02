import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        # TODO Create necessary layers
        self.input_width, self.input_height, self.input_channels = input_shape

        self.layers = [
            ConvolutionalLayer(in_channels=self.input_channels, out_channels=conv1_channels, filter_size=3, padding=1),
            ReLULayer(),
            MaxPoolingLayer(4, 4),  # 32x32 -> 8x8
            ConvolutionalLayer(in_channels=conv1_channels, out_channels=conv2_channels, filter_size=3, padding=1),
            ReLULayer(),
            MaxPoolingLayer(4, 4),  # 8x8 -> 2x2
            Flattener(),
            FullyConnectedLayer(4*conv2_channels, n_output_classes) # 2 x 2 x conv2
        ]

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # TODO Compute loss and fill param gradients

        for param in self.params().values():
            param.grad.fill(0.0)

        fwd = X.copy()
        for layer in self.layers:
            fwd = layer.forward(fwd)

        loss, grad = softmax_with_cross_entropy(fwd, y)

        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return loss

    def predict(self, X):
        fwd = X.copy()
        for layer in self.layers:
            fwd = layer.forward(fwd)

        return np.argmax(fwd, axis = 1)


    def params(self):
        # TODO: Aggregate all the params from all the layers
        # which have parameters
        result = {
            'FConvW' : self.layers[0].params()['W'],
            'FConvB' : self.layers[0].params()['B'],
            'SConvW' : self.layers[3].params()['W'],
            'SConvB' : self.layers[3].params()['B'],
            'FCW' : self.layers[7].params()['W'],
            'FCB' : self.layers[7].params()['B']
        }

        return result
