import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


class Perceptron:
    """
    The perceptron is a single unit of a neural network. This perceptron specifically is meant to be used by itself
    to approximate linear or nonlinear functions. The formulae here should not be used inside a neural network.
    """

    def __init__(self, input_dim: int, lr: float = 0.1):
        self.input_dim = input_dim
        self.weights: np.ndarray = np.random.random(input_dim + 1) * 0.1  # dim + 1 for the bias term
        self.lr = lr

    def forward(self, x: np.ndarray, threshold=True):
        x = np.append(x, 1)  # append the bias term
        o = np.sum(self.weights * x)
        o = sigmoid(o)
        o[o < 0] = -1
        o[o >= 0] = 1
        return o

    def update_perceptron_rule(self, x: np.ndarray, t: np.ndarray):
        """
        The perceptron update rule is only useful when the target solution is linearly separable
        """
        y = self.forward(x)
        delta_w = self.lr * (t - y) * x
        self.weights += delta_w

    def update_gradient_descent(self, x: np.ndarray, t: np.ndarray):
        """
        Gradient descent (aka the delta rule) can be used to update weights when the target space is not
        linearly separable, computing weight updates after summing over all of the training examples
        """
        x = np.append(x, 1)
        o = np.sum(self.weights * x)
        dw = np.zeros(self.weights.shape)
        for xi, ti in zip(x, t):
            o = np.sum(self.weights * xi)
            dw = dw + self.lr * (ti - o) * xi  # the delta ruler, or LMS rule

        self.weights += dw

    def update_sgd(self, x: np.ndarray, t: np.ndarray):
        """
        Stochastic approximation of the gradient descent update rule, to alleviate the following problems:
        1. converging to a local minimum can sometimes be quite slow (i.e. many thousands of gradient descent steps)
        2. if the are multiple local minima in the error surface, then there is no guarantee that the procedure will
        find the global minimum

        It does this by updating weights incrementally, following the calculation of the error for each individual
        example
        """
        x = np.append(x, 1)
        y = self.forward(x)
        dw = np.zeros(self.weights.shape)
        for xi, ti in zip(x, t):
            o = np.sum(self.weights * xi)
            self.weights += dw + self.lr * (ti - o) * xi