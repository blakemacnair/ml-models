from sklearn.metrics import plot_precision_recall_curve, plot_roc_curve, plot_confusion_matrix, precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.model_selection import learning_curve

import matplotlib.pyplot as plt
from typing import Dict, List, Any
from sklearn.base import ClassifierMixin
from sklearn.model_selection import ShuffleSplit
import numpy as np


def compare_models(classifiers: Dict[str, ClassifierMixin], cv: ShuffleSplit,
                   x: np.ndarray, y: np.ndarray, validation_size=0.2):
    train_scores: Dict[str, List] = {}
    test_scores: Dict[str, List] = {}

    for name in classifiers.keys():
        train_scores[name] = []
        test_scores[name] = []

    validation_cv = ShuffleSplit(n_splits=1, test_size=validation_size, random_state=0)
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


def plot_compare_learning_curve(classifiers: Dict[str, Any], x: np.ndarray, y: np.ndarray,
                                cv=ShuffleSplit(n_splits=5),
                                train_sizes=np.linspace(0.1, 1.0, 9)):
    fig = plt.figure()
    ax = fig.add_subplot()

    ax.set_title('Learning curves')
    ax.set_xlabel('Sample size')
    ax.set_ylabel('Accuracy')

    for name, clf in classifiers.items():
        train_sizes, train_scores, test_scores = learning_curve(clf, x, y, cv=cv,
                                                                train_sizes=train_sizes,
                                                                random_state=0)
        test_scores_mean = np.mean(test_scores, axis=1)
        ax.plot(train_sizes, test_scores_mean, 'o-', label=name)

    ax.legend()
    plt.show()


def plot_compare_roc_curve(classifiers: Dict[str, Any], x: np.ndarray, y: np.ndarray, validation_size=0.2):
    validation_cv = ShuffleSplit(n_splits=1, test_size=validation_size, random_state=0)
    train_ind, validation_ind = validation_cv.split(x, y).__next__()

    x_validation, y_validation = x[validation_ind], y[validation_ind]
    x, y = x[train_ind], y[train_ind]

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title('ROC curves')

    for name, clf in classifiers.items():
        clf.fit(x, y)
        plot_roc_curve(clf, x_validation, y_validation, name=name, ax=ax)

    ax.legend()
    plt.show()


def plot_compare_precision_recall_curve(classifiers: Dict[str, Any], x: np.ndarray, y: np.ndarray, validation_size=0.2):
    validation_cv = ShuffleSplit(n_splits=1, test_size=validation_size, random_state=0)
    train_ind, validation_ind = validation_cv.split(x, y).__next__()

    x_validation, y_validation = x[validation_ind], y[validation_ind]
    x, y = x[train_ind], y[train_ind]

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title('Precision-Recall curves')

    for name, clf in classifiers.items():
        clf.fit(x, y)
        plot_precision_recall_curve(clf, x_validation, y_validation, name=name, ax=ax)

    ax.legend()
    plt.show()
