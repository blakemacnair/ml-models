from sklearn.metrics import plot_precision_recall_curve, plot_roc_curve, plot_confusion_matrix
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from typing import Dict, List
from sklearn.base import ClassifierMixin
from sklearn.model_selection import ShuffleSplit
import numpy as np


def compare_models(classifiers: Dict[str, ClassifierMixin], cv: ShuffleSplit,
                   x: np.ndarray, y: np.ndarray, draw_plots=True, validation_size=0.2):
    train_scores: Dict[str, List] = {}
    test_scores: Dict[str, List] = {}

    for name in classifiers.keys():
        train_scores[name] = []
        test_scores[name] = []

    validation_cv = ShuffleSplit(n_splits=1, test_size=validation_size)
    train_ind, validation_ind = validation_cv.split(x, y).__next__()

    x_validation, y_validation = x[validation_ind], y[validation_ind]
    x, y = x[train_ind], y[train_ind]

    for train_ind, test_ind in cv.split(x, y):
        x_train, y_train = x[train_ind], y[train_ind]
        x_test, y_test = x[test_ind], y[test_ind]

        for name, clf in classifiers.items():
            clf.fit(x_train, y_train)
            train_scores[name].append(clf.score(x_train, y_train))
            test_scores[name].append(clf.score(x_test, y_test))

    for name, clf in classifiers.items():
        plt.figure()
        ax = plt.subplot(2, 2, 1)
        disp = plot_precision_recall_curve(clf, x_validation, y_validation, ax=ax)
        disp.ax_.set_title('{} Precision-Recall curve'.format(name))
        ax = plt.subplot(2, 2, 2)
        disp = plot_roc_curve(clf, x_validation, y_validation, ax=ax)
        disp.ax_.set_title('{} ROC curve'.format(name))
        ax = plt.subplot(2, 2, 3)
        disp = plot_confusion_matrix(clf, x_validation, y_validation, ax=ax)
        disp.ax_.set_title('{} Confusion matrix curve'.format(name))
        plt.show()

    return train_scores, test_scores
