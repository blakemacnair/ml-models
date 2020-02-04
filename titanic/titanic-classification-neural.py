from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import ShuffleSplit

from NeuralNet import MurderBot
from titanic.titanic_dataset import import_cleaned_titanic_data

from metrics import plot_compare_precision_recall_curve, plot_compare_roc_curve, plot_compare_learning_curve


if __name__ == "__main__":
    x, y, x_test, test_ids = import_cleaned_titanic_data()

    x = Tensor(x)

    x_test = Tensor(x_test)

    y = y.reshape(-1, 1)
    y = Tensor(np.concatenate((y, 1 - y), axis=1))

    mbot = MurderBot(input_dim=x.shape[1],
                     hidden_dims=[int(x.shape[1] * 1.25), int(x.shape[1] / 2)],
                     output_dim=y.shape[1])

    classifiers = {'MurderBot': mbot}
    plot_compare_roc_curve(classifiers, x, y)

    cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
    # for train_index, test_index in cv.split(X, y):
    #     train_x = X[train_index]
    #     train_y = y[train_index]
    #
    #     test_x = X[test_index]
    #     test_y = y[test_index]
    #
    #     train_dataset = TensorDataset(train_x, train_y)
    #     train_loader = DataLoader(dataset=train_dataset, batch_size=10)
    #
    #     for epoch in range(3):
    #         mbot.fit(train_x, train_y)
    #
    #     test_p = mbot.predict(test_x)
    #     score = float((test_p == test_y).sum()) / (test_p.shape[0] * test_p.shape[1])
    #     print("Score: {}".format(score))
    print("Done boiiiii")
