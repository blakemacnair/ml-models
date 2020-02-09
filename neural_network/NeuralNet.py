from torch import nn
import numpy as np

from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring

from collections import OrderedDict

from sklearn.datasets import make_classification
from metrics import plot_compare_roc_curve, plot_compare_precision_recall_curve, plot_compare_learning_curve


class MurderBot(nn.Module):
    def __init__(self, input_size=28, hidden_layers: np.ndarray = np.array([100, 30]), output_size=2):
        super(MurderBot, self).__init__()

        modules = OrderedDict()

        prev_layer_size = input_size
        for layer, layer_size in enumerate(hidden_layers):
            modules['linear-{}'.format(layer)] = nn.Linear(prev_layer_size, layer_size)
            modules['ReLU-{}'.format(layer)] = nn.ReLU()
            prev_layer_size = layer_size

        modules['output'] = nn.Linear(prev_layer_size, output_size)
        modules['softmax-output'] = nn.Softmax(dim=-1)

        self.model = nn.Sequential(modules)

    def forward(self, x):
        return self.model(x)


def skorch_murder_bot(input_size=28, hidden_layers: np.ndarray = np.array([100, 30]), batch_size=5):
    auc = EpochScoring(scoring='roc_auc', lower_is_better=False)
    return NeuralNetClassifier(
        MurderBot,
        module__input_size=input_size,
        module__hidden_layers=hidden_layers,
        max_epochs=30,
        batch_size=batch_size,
        lr=0.1,
        callbacks=[auc],
        verbose=False
    )


if __name__ == "__main__":
    x, y = make_classification(2000, 40, n_informative=10, random_state=0)
    x = x.astype(np.float32)

    input_size = x.shape[1]
    nets = {
        '3 hidden': skorch_murder_bot(input_size=input_size, hidden_layers=np.array([input_size * 3,
                                                                                     input_size * 2,
                                                                                     input_size // 2])),
        '2 hidden': skorch_murder_bot(input_size=input_size, hidden_layers=np.array([input_size * 2,
                                                                                     input_size // 2])),
        '1 hidden': skorch_murder_bot(input_size=input_size, hidden_layers=np.array([input_size * 2])),
        '0 hidden': skorch_murder_bot(input_size=input_size, hidden_layers=np.array([]))
    }

    for i in range(input_size * 2, input_size * 3, input_size // 2):
        for j in range(input_size, input_size * 2, input_size // 2):
            for k in range(input_size // 2, input_size, input_size // 3):
                nets['{}-{}-{}'.format(i, j, k)] = skorch_murder_bot(input_size=input_size,
                                                                     hidden_layers=np.array([input_size * 3,
                                                                                             input_size * 2,
                                                                                             input_size // 2]))
    plot_compare_roc_curve(nets, x, y)
    plot_compare_precision_recall_curve(nets, x, y)
    # plot_compare_learning_curve(nets, x, y, train_sizes=np.linspace(0.3, 1.0, 5))
