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


def plot_learning_curves(classifiers: Dict[str, ClassifierMixin], cv: ShuffleSplit,
                         x: np.ndarray, y: np.ndarray):
    for name, clf in classifiers.items():
        plt.figure()

        # Get learning data
        train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(clf, x, y, cv=cv,
                                                                                        return_times=True,
                                                                                        random_state=0)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)

        # Plot learning curve
        ax_1 = plt.subplot(2, 2, 1)
        ax_1.grid()
        ax_1.fill_between(train_sizes, train_scores_mean - train_scores_std,
                          train_scores_mean + train_scores_std, alpha=0.1,
                          color="r")
        ax_1.fill_between(train_sizes, test_scores_mean - test_scores_std,
                          test_scores_mean + test_scores_std, alpha=0.1,
                          color="g")
        ax_1.plot(train_sizes, train_scores_mean, 'o-', color="r",
                  label="Training score")
        ax_1.plot(train_sizes, test_scores_mean, 'o-', color="g",
                  label="Cross-validation score")
        ax_1.legend(loc="best")

        # Plot n_samples vs fit_times
        ax_2 = plt.subplot(2, 2, 2)
        ax_2.grid()
        ax_2.plot(train_sizes, fit_times_mean, 'o-')
        ax_2.fill_between(train_sizes, fit_times_mean - fit_times_std,
                          fit_times_mean + fit_times_std, alpha=0.1)
        ax_2.set_xlabel("Training examples")
        ax_2.set_ylabel("fit_times")
        ax_2.set_title("Scalability of the model")

        # Plot fit_time vs score
        ax_3 = plt.subplot(2, 2, 3)
        ax_3.grid()
        ax_3.plot(fit_times_mean, test_scores_mean, 'o-')
        ax_3.fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                          test_scores_mean + test_scores_std, alpha=0.1)
        ax_3.set_xlabel("fit_times")
        ax_3.set_ylabel("Score")
        ax_3.set_title("Performance of the model")

        plt.show()


def plot_compare_learning_curve(classifiers: Dict[str, Any], x: np.ndarray, y: np.ndarray, cv: ShuffleSplit,
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
