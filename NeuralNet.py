import numpy as np
import pandas as pd


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


class ANN:
    def __init__(self, input_dims: int, output_dims: int, hidden_dims: int,
                 learning_rate: float = 0.1, decay: float = 0, momentum: float = 0):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims

        self.w_h = (np.random.rand(hidden_dims, input_dims) - 0.5) * 0.1
        self.w_k = (np.random.rand(output_dims, hidden_dims) - 0.5) * 0.1

        assert(1 >= learning_rate >= 0)
        self.lr = learning_rate

        assert(1 >= decay >= 0)
        self.decay = decay

        assert (1 >= momentum >= 0)
        self.momentum = momentum

    def forward(self, x):
        o_h = sigmoid(self.w_h @ x)
        o = sigmoid(self.w_k @ o_h)
        return o

    def backpropagation(self, x: np.ndarray, t: np.ndarray):
        """
        For each output unit k, calculate its error as d_k = o_k * (1 - o_k) * (t_k - o_k)
        where o_k * (1 - o_k) comes from the derivative of the sigmoid

        For each hidden unit h, calculate its error as d_h = x_kh * (1 - x_kh) * sum(w_kh * d_k, for k in output layer)

        Update network weight wji by w_ji += lr * d_j * x_ji
        :return:
        """
        x = x.reshape((len(x), 1))
        t = t.reshape((len(t), 1))

        x_kh = sigmoid(self.w_h @ x)
        o = sigmoid(self.w_k @ x_kh)

        d_k = o * (1 - o) * (t - o)
        d_h = x_kh * (1 - x_kh) * (d_k.transpose() @ self.w_k).transpose()

        dx_kh = np.repeat(x_kh, self.output_dims, axis=-1).transpose()
        dx = np.repeat(x, self.hidden_dims, axis=-1).transpose()

        dw_h = d_h * dx
        dw_k = d_k * dx_kh

        self.w_h += self.lr * dw_h
        self.w_k += self.lr * dw_k


if __name__ == "__main__":
    d_in = 4
    d_out = 2
    d_h = 10
    net = ANN(d_in, d_out, d_h)
    x = np.random.random(d_in)
    y = np.random.randint(0, 2, size=2)
    net.backpropagation(x, y)
