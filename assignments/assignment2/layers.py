from curses import delay_output
from pyexpat.errors import XML_ERROR_DUPLICATE_ATTRIBUTE
import numpy as np


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
    
    loss = reg_strength * np.sum(np.square(W))
    grad = reg_strength * 2 * W

    return loss, grad


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''

    if(predictions.ndim == 1):
      exponents = np.exp(predictions - np.max(predictions))
      return exponents / np.sum(exponents)

    exponents = np.exp(predictions - np.amax(predictions, axis=1).reshape(-1, 1))
    devs = np.sum(exponents, axis = 1).reshape(-1, 1)
    return exponents / devs

def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''

    if(probs.ndim == 1):
      return (-np.log(probs[target_index]))

    batch_size = target_index.size
    idxs = np.arange(target_index.size)
    return -np.sum(np.log(probs[idxs, target_index])) / batch_size

def softmax_with_cross_entropy(preds, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    # TODO: Copy from the previous assignment
    
    probs = softmax(preds)
    loss = cross_entropy_loss(probs, target_index)
    dprediction = probs
    if (probs.ndim == 1):
        dprediction[target_index] -= 1
    else:
        batch_size = target_index.size
        idxs = np.arange(batch_size)
        dprediction[idxs, target_index] -= 1
        dprediction /= batch_size

    return loss, dprediction



class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    fwd_data = np.zeros(1)

    def __init__(self):
        pass

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass

      self.fwd_data = X
      # X[X < 0] = 0
      # return X
      return np.where(X < 0, 0, X)


    def backward(self, d_out):
      """
      Backward pass

      Arguments:
      d_out, np array (batch_size, num_features) - gradient
          of loss function with respect to output

      Returns:
      d_result: np array (batch_size, num_features) - gradient
        with respect to input
      """
      # TODO: Implement backward pass
      # Your final implementation shouldn't have any loops
      
      return np.where(self.fwd_data < 0, 0, 1) * d_out


    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
      
      self.X = X
      return X.dot(self.W.value) + self.B.value

    def backward(self, d_out):
      """
      Backward pass
      Computes gradient with respect to input and
      accumulates gradients within self.W and self.B

      Arguments:
      d_out, np array (batch_size, n_output) - gradient
          of loss function with respect to output

      Returns:
      d_result: np array (batch_size, n_input) - gradient
        with respect to input
      """
      # TODO: Implement backward pass
      # Compute both gradient with respect to input
      # and gradients with respect to W and B
      # Add gradients of W and B to their `grad` attribute

      # It should be pretty similar to linear classifier from
      # the previous assignment

      d_input = d_out.dot(self.W.value.T)
      self.W.grad += self.X.T.dot(d_out)
      self.B.grad += np.ones((1, d_out.shape[0])).dot(d_out)
      
      return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}
