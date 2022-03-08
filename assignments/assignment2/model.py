import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        
        self.struct = [FullyConnectedLayer(n_input, hidden_layer_size),
                        ReLULayer(),
                        FullyConnectedLayer(hidden_layer_size, n_output)]


    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!

        for _, elem in self.params().items():
          elem.grad = 0

        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!

        dat = X.copy()
        for elem in self.struct:
          dat = elem.forward(dat)

        loss, grad = softmax_with_cross_entropy(dat, y)

        for elem in reversed(self.struct):
          grad = elem.backward(grad)

        for _, elem in self.params().items():
          l2_loss, l2_grad = l2_regularization(elem.value, self.reg)
          elem.grad += l2_grad
          loss += l2_loss

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)

        for _, elem in self.params().items():
          elem.grad = 0

        dat = X.copy()
        for elem in self.struct:
          dat = elem.forward(dat)

        pred = dat.argmax(axis=1)
        return pred

    def params(self):
        # TODO Implement aggregating all of the params

      return { 'W_1' : self.struct[0].params()['W'],
                   'B_1' : self.struct[0].params()['B'],
                   'W_2' : self.struct[2].params()['W'],
                   'B_2' : self.struct[2].params()['B'],
        }