from torch import nn
import numpy as np

from skorch import NeuralNetClassifier, NeuralNetBinaryClassifier
from skorch.toy import MLPModule
from skorch.callbacks import EpochScoring

from collections import OrderedDict

from credit_card_fraud.dataset import load_normalized_credit_fraud_numpy, load_credit_fraud_numpy
from online_shopper_intention.dataset import load_normalized_shopper_intention_numpy
from sklearn.model_selection import train_test_split

from metrics import compare_models_all_metrics


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
    # Load the data
    x_cr, y_cr = load_normalized_credit_fraud_numpy(filepath='../data/creditcard.csv')
    x_sh, y_sh = load_normalized_shopper_intention_numpy(filepath='../data/online_shoppers_intention.csv')
    train_sizes = np.linspace(0.3, 0.6, 4)

    x_sh = x_sh.astype(np.float32)
    input_size = x_sh.shape[1]
    nets = {
        'murderbot': skorch_murder_bot(input_size=input_size,
                                       hidden_layers=np.array([input_size * 2, input_size // 2])),
    }
    # compare_models_all_metrics(nets, x_sh, y_sh, train_sizes)

    x_cr = x_cr.astype(np.float32)
    y_cr = y_cr.astype(np.int)
    input_size = x_cr.shape[1]
    nets = {
        'murderbot': skorch_murder_bot(input_size=input_size,
                                       hidden_layers=np.array([input_size * 2, input_size // 2])),
    }
    compare_models_all_metrics(nets, x_cr, y_cr, train_sizes)
