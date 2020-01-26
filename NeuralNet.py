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

        self.prev_dw_h = None
        self.prev_dw_k = None

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

        dw_h = self.lr * d_h * dx
        dw_k = self.lr * d_k * dx_kh

        self.w_h += dw_h
        self.w_k += dw_k

        if self.decay > 0:
            if self.prev_dw_h is not None:
                m_h = self.decay * self.prev_dw_h
                self.w_h += m_h

            if self.prev_dw_h is not None:
                m_k = self.decay * self.prev_dw_k
                self.w_k += m_k

            self.prev_dw_h = dw_h
            self.prev_dw_k = dw_k

        err = np.sum(np.square(self.forward(x).flatten() - y.flatten()))
        delta = (np.mean(dw_h) + np.mean(dw_k)) / 2
        

        return err, delta

    def train(self, x, y, max_iter=200, min_delta_error=1e-5):
        delta_error = np.inf
        last_err = np.inf

        i = 0

        while delta_error > min_delta_error and i < max_iter:
            err, delta = self.backpropagation(x, y)
            delta_error = np.abs(last_err - err)
            last_err = err
            i += 1
            print(delta_error)


if __name__ == "__main__":
    d_in = 7
    d_out = 4
    d_h = 25
    net = ANN(d_in, d_out, d_h)
    x = np.random.random(d_in)
    y = np.random.randint(0, 2, size=d_out)

    net.train(x, y)
