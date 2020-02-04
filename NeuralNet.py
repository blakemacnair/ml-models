from torch import nn, Tensor
import numpy as np

from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring

from sklearn.datasets import make_classification
from titanic.titanic_dataset import import_cleaned_titanic_data
from metrics import plot_compare_roc_curve, plot_compare_precision_recall_curve, plot_compare_learning_curve


class MurderBot(nn.Module):
    def __init__(self):
        super(MurderBot, self).__init__()

        input_dim = 26
        output_dim = 2
        self.model = nn.Sequential(
            nn.Linear(input_dim, 30),
            nn.ReLU(),
            nn.Linear(30, 10),
            nn.ReLU(),
            nn.Linear(10, output_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.model(x)


def skorch_murder_bot():
    auc = EpochScoring(scoring='roc_auc', lower_is_better=False)
    return NeuralNetClassifier(
        MurderBot,
        max_epochs=20,
        lr=0.1,
        iterator_train__shuffle=True,
        callbacks=[auc],
    )


if __name__ == "__main__":
    # X, Y = make_classification(2000, 40, n_informative=10, random_state=0)
    # X = X.astype(np.float32)

    x, y, x_test, test_ids = import_cleaned_titanic_data(directorypath='titanic/')

    x = x.astype(np.float32)
    y = y.astype(np.int)

    net = skorch_murder_bot()
    plot_compare_roc_curve({'MurderBot': net}, x, y)
    # plot_compare_learning_curve({'MurderBot': net}, x, y)
    plot_compare_precision_recall_curve({'MurderBot': net}, x, y)

