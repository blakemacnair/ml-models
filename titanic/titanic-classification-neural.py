import numpy as np
import random

from neural_network.NeuralNet import skorch_murder_bot
from titanic.titanic_dataset import import_cleaned_titanic_data

from metrics import plot_compare_precision_recall_curve, plot_compare_roc_curve

if __name__ == "__main__":
    random.seed(0)
    x, y, x_test, test_ids = import_cleaned_titanic_data()

    x = x.astype(np.float32)
    y = y.astype(np.int)

    input_size = x.shape[1]
    layers_classifiers = {
        '3-2-1/2': skorch_murder_bot(input_size=input_size, hidden_layers=np.array([input_size * 3,
                                                                                    input_size * 2,
                                                                                    input_size // 2])),
        '2-2'    : skorch_murder_bot(input_size=input_size, hidden_layers=np.array([input_size * 2,
                                                                                    input_size * 2])),
        '3/2'    : skorch_murder_bot(input_size=input_size, hidden_layers=np.array([input_size * 3 // 2])),
        '2'      : skorch_murder_bot(input_size=input_size, hidden_layers=np.array([input_size * 2])),
    }
    plot_compare_roc_curve(layers_classifiers, x, y)
    plot_compare_precision_recall_curve(layers_classifiers, x, y)

    hidden_layers = np.array([input_size * 2, input_size * 2])
    batches_classifiers = {
        '3'  : skorch_murder_bot(input_size=input_size, hidden_layers=hidden_layers, batch_size=3),
        '15' : skorch_murder_bot(input_size=input_size, hidden_layers=hidden_layers, batch_size=15),
        '50' : skorch_murder_bot(input_size=input_size, hidden_layers=hidden_layers, batch_size=50),
        '100': skorch_murder_bot(input_size=input_size, hidden_layers=hidden_layers, batch_size=100)
    }
    plot_compare_roc_curve(batches_classifiers, x, y)
    plot_compare_precision_recall_curve(batches_classifiers, x, y)
