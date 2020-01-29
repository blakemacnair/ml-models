import torch
from torch import Tensor, nn
from torch.optim import SGD
from torch.optim.optimizer import Optimizer
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import OneHotEncoder

import matplotlib.pyplot as plt


class MurderBot(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MurderBot, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        hidden_dim = int(input_dim / 2)

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        return self.model(x).round()

    def fit(self, data_loader: DataLoader, loss_func, optimizer):
        loss_history = []
        acc_history = []
        for x_batch, y_batch in data_loader:

            self.train()

            pred = self(x_batch)
            loss = loss_func(pred, y_batch)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            self.eval()

            accuracy = float((self.predict(x_batch) == y_batch).sum()) / (pred.shape[0] * pred.shape[1])
            loss_history.append(loss)
            acc_history.append(accuracy)
        return loss_history, acc_history


def mushrooms_to_numeric(s: pd.DataFrame):
    enc = OneHotEncoder(drop='first')
    enc = enc.fit(s)

    s_enc = enc.transform(s).toarray()
    return s_enc, enc


def import_mushrooms() -> pd.DataFrame:
    return pd.read_csv("mushrooms.csv")


if __name__ == "__main__":
    s = import_mushrooms()

    s_enc, enc = mushrooms_to_numeric(s)
    s_enc = Tensor(s_enc)
    X = s_enc[:, 1:]
    y = s_enc[:, 0].reshape(-1, 1)
    y = Tensor(np.concatenate((y, 1 - y), axis=1))

    cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
    epoch = 0
    for train_index, test_index in cv.split(X, y):
        mbot = MurderBot(input_dim=X.shape[1], output_dim=y.shape[1])
        optimizer = SGD(mbot.parameters(), lr=0.1)
        criterion = nn.MSELoss()

        train_x = X[train_index]
        train_y = y[train_index]

        test_x = X[test_index]
        test_y = y[test_index]

        train_dataset = TensorDataset(train_x, train_y)
        train_loader = DataLoader(dataset=train_dataset, batch_size=100)

        loss, acc = mbot.fit(train_loader, criterion, optimizer)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.set_title("Loss")
        ax1.plot(loss)
        ax2.set_title("Accuracy")
        ax2.plot(acc)
        plt.show()

        test_p = mbot.predict(test_x)
        score = float((test_p == test_y).sum()) / (test_p.shape[0] * test_p.shape[1])
        print("Score: {}".format(score))

        epoch += 1
    print("Done boiiiii")
