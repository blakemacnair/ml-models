from sklearn.metrics import plot_precision_recall_curve, plot_roc_curve, plot_confusion_matrix, precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import get_scorer

from sklearn.model_selection import learning_curve
from sklearn.base import clone

import matplotlib.pyplot as plt
from typing import Dict, List, Any
from sklearn.base import ClassifierMixin
from sklearn.model_selection import ShuffleSplit
import numpy as np
from scipy.interpolate import make_interp_spline


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
                                train_sizes=np.linspace(0.1, 1.0, 9),
                                scoring='accuracy'):
    fig = plt.figure()
    ax = fig.add_subplot()

    ax.set_title('Learning curves')
    ax.set_xlabel('Sample size')
    ax.set_ylabel('Accuracy')

    for name, clf in classifiers.items():
        train_sizes, train_scores, test_scores = learning_curve(clf, x, y, cv=cv,
                                                                train_sizes=train_sizes,
                                                                random_state=0,
                                                                scoring=get_scorer(scoring))
        test_scores_mean = np.mean(test_scores, axis=1)

        spl = make_interp_spline(train_sizes, test_scores_mean, k=3)

        x_new = np.linspace(train_sizes.min(), train_sizes.max(), 200)
        y_new = spl(x_new).T
        ax.plot(x_new, y_new, label=name)

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


def plot_cost_complexity_pruning_path(tree, x, y):
    path = tree.cost_complexity_pruning_path(x, y)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    fig, ax = plt.subplots()
    ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
    ax.set_xlabel("effective alpha")
    ax.set_ylabel("total impurity of leaves")
    ax.set_title("Total Impurity vs effective alpha for training set")
    plt.show()


def plot_nodes_vs_alpha(tree, x, y):
    path = tree.cost_complexity_pruning_path(x, y)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    trees = []
    for ccp_alpha in ccp_alphas:
        tree1 = clone(tree)
        tree1.ccp_alpha = ccp_alpha
        tree1.fit(x, y)
        trees.append(tree1)

    trees = trees[:-1]
    ccp_alphas = ccp_alphas[:-1]

    node_counts = [clf.tree_.node_count for clf in trees]
    depth = [clf.tree_.max_depth for clf in trees]
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(ccp_alphas, node_counts, marker='o', drawstyle="steps-post")
    ax[0].set_xlabel("alpha")
    ax[0].set_ylabel("number of nodes")
    ax[0].set_title("Number of nodes vs alpha")
    ax[1].plot(ccp_alphas, depth, marker='o', drawstyle="steps-post")
    ax[1].set_xlabel("alpha")
    ax[1].set_ylabel("depth of tree")
    ax[1].set_title("Depth vs alpha")
    fig.tight_layout()
    plt.show()


def plot_acc_alpha_train_vs_test(tree, x_train, y_train, x_test, y_test):
    path = tree.cost_complexity_pruning_path(x_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    trees = []
    for ccp_alpha in ccp_alphas:
        clf1 = clone(tree)
        clf1.ccp_alpha = ccp_alpha
        clf1.fit(x_train, y_train)
        trees.append(clf1)

    train_scores = [clf.score(x_train, y_train) for clf in trees]
    test_scores = [clf.score(x_test, y_test) for clf in trees]

    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs alpha for training and testing sets")
    ax.plot(ccp_alphas, train_scores, marker='o', label="train", drawstyle="steps-post")
    ax.plot(ccp_alphas, test_scores, marker='o', label="test", drawstyle="steps-post")
    ax.legend()
    plt.show()
