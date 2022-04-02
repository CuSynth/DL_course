import numpy as np


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

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

def softmax_with_cross_entropy(predictions, target_index):
    '''
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
    '''

    probs = softmax(predictions)
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
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        self.fwd_data = X
        return np.where(X < 0, 0, X)

    def backward(self, d_out):
        return np.where(self.fwd_data < 0, 0, 1) * d_out

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
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

      d_input = d_out.dot(self.W.value.T)
      self.W.grad += self.X.T.dot(d_out)
      self.B.grad += np.ones((1, d_out.shape[0])).dot(d_out)
      
      return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding


    def forward(self, X):
        batch_size, height, width, channels = X.shape

        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below

        out_height = height + 2 * self.padding - self.filter_size + 1
        out_width  = width  + 2 * self.padding - self.filter_size + 1        

        self.X = np.pad(X, (
            (0, 0),
            (self.padding, self.padding),
            (self.padding, self.padding),
            (0, 0)
        ), 'constant')

        W_flat = self.W.value.reshape(self.in_channels * self.filter_size**2, self.out_channels)
        res = np.zeros((batch_size, out_height, out_width, self.out_channels))
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement forward pass for specific location
                X_flat = self.X[:, y:y+self.filter_size, x:x+self.filter_size, :]
                X_flat = X_flat.reshape(batch_size, self.in_channels * self.filter_size**2)
                res[:, y, x, :] = np.dot(X_flat, W_flat) + self.B.value

        return res


    def backward(self, d_out):
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        dX =  np.zeros_like(self.X)
        W_flat = self.W.value.reshape(self.in_channels * self.filter_size**2, self.out_channels)

        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                X_flat = self.X[:, y:y+self.filter_size , x:x+self.filter_size, :]           
                X_flat = X_flat.reshape(batch_size, self.in_channels * self.filter_size**2)

                dX[:, y:y+self.filter_size, x:x+self.filter_size, :] += np.dot(d_out[:, y, x, :], W_flat.T).reshape(batch_size, self.filter_size, self.filter_size, self.in_channels)
                self.W.grad += np.dot(X_flat.T, d_out[:, y, x, :]).reshape(self.filter_size, self.filter_size, self.in_channels, out_channels)

        self.B.grad = np.sum(d_out, axis=tuple(range(len(d_out.shape)-1))).reshape(out_channels)

        if(self.padding):
            dX = dX[:, self.padding:-self.padding, self.padding:-self.padding, :]

        return dX
    
    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        # TODO: Implement maxpool forward pass
        batch_size, height, width, channels = X.shape

        self.X = X.copy()

        out_height = height // self.pool_size
        out_width  = width  // self.pool_size
        res = np.zeros((batch_size, out_height, out_width, channels))

        for y in range(out_height):
            for x in range(out_width):
                X_flat = X[:, y:y+self.pool_size, x:x+self.pool_size, :]
                res[:, y, x, :] = np.amax(X_flat, axis=(1, 2))

        return  res

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape

        out_height = height // self.pool_size
        out_width  = width  // self.pool_size
        res = np.zeros_like(self.X)

        for y in range(out_height):
            for x in range(out_width):
                X_flat = self.X[:, y:y+self.pool_size, x:x+self.pool_size, :]
                grad = d_out[:, y, x, :][:, np.newaxis, np.newaxis, :]
                mask = (X_flat == np.amax(X_flat, (1, 2))[:, np.newaxis, np.newaxis, :])
                res[:, y:y+self.pool_size, x:x+self.pool_size, :] += grad * mask

        return res

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X_shape = X.shape

        # TODO: Implement forward pass
        return X.reshape(batch_size, height*width*channels)

    def backward(self, d_out):
        # TODO: Implement backward pass
        return d_out.reshape(self.X_shape)

    def params(self):
        # No params!
        return {}
