from sklearn.metrics import plot_precision_recall_curve, plot_roc_curve, plot_confusion_matrix
from sklearn.metrics import get_scorer, precision_recall_curve

from sklearn.model_selection import learning_curve

import matplotlib.pyplot as plt
from typing import Dict, List, Any
from sklearn.base import ClassifierMixin
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
import numpy as np
from scipy.interpolate import interp1d


def compare_models_all_metrics(models,
                               x,
                               y,
                               train_sizes,
                               scoring='balanced_accuracy',
                               k_folds=5,
                               title_prefix=None):
    plot_compare_roc_curve(models, x, y, title_prefix=title_prefix)
    plot_compare_precision_recall_curve(models, x, y, title_prefix=title_prefix)
    plot_compare_learning_curve(models, x, y, k_folds=k_folds, train_sizes=train_sizes, scoring=scoring,
                                title_prefix=title_prefix)


def compare_models(classifiers: Dict[str, ClassifierMixin], cv: StratifiedShuffleSplit,
                   x: np.ndarray, y: np.ndarray, validation_size=0.2):
    train_scores: Dict[str, List] = {}
    test_scores: Dict[str, List] = {}

    for name in classifiers.keys():
        train_scores[name] = []
        test_scores[name] = []

    validation_cv = StratifiedShuffleSplit(n_splits=1, test_size=validation_size, random_state=0)
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
                                k_folds=5,
                                train_sizes=np.linspace(0.1, 1.0, 9),
                                scoring='balanced_accuracy',
                                title_prefix=None):
    """
    scoring param can be ['balanced_accuracy', 'precision', 'auc_roc'] among others
    """
    # TODO: Use Seaborn instead of vanilla pyplot
    fig = plt.figure()
    ax = fig.add_subplot()

    title = 'Learning curves'
    if title_prefix is not None:
        title = title_prefix + ": " + title
    ax.set_title(title)
    ax.set_xlabel('Sample size')
    ax.set_ylabel(scoring)

    for name, clf in classifiers.items():
        train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(clf, x, y,
                                                                                        cv=k_folds,
                                                                                        train_sizes=train_sizes,
                                                                                        random_state=0,
                                                                                        scoring=get_scorer(scoring),
                                                                                        return_times=True,
                                                                                        n_jobs=-1)
        test_scores_mean = np.mean(test_scores, axis=1)

        spl = interp1d(train_sizes, test_scores_mean, kind='linear')
        train_sizes_spaced = np.linspace(train_sizes.min(), train_sizes.max(), 200)
        interp_scores = spl(train_sizes_spaced).T
        ax.plot(train_sizes_spaced, interp_scores, label=name)

    ax.legend()
    plt.show()


def plot_compare_roc_curve(classifiers: Dict[str, Any], x: np.ndarray, y: np.ndarray, validation_size=0.2,
                           title_prefix=None):
    validation_cv = StratifiedShuffleSplit(n_splits=1, test_size=validation_size, random_state=0)
    train_ind, validation_ind = validation_cv.split(x, y).__next__()

    x_validation, y_validation = x[validation_ind], y[validation_ind]
    x, y = x[train_ind], y[train_ind]

    fig = plt.figure()
    ax = fig.add_subplot()
    title = 'ROC Curves'
    if title_prefix is not None:
        title = title_prefix + ": " + title
    ax.set_title(title)

    for name, clf in classifiers.items():
        clf.fit(x, y)
        plot_roc_curve(clf, x_validation, y_validation, name=name, ax=ax)

    ax.legend()
    plt.show()


def plot_compare_precision_recall_curve(classifiers: Dict[str, Any], x: np.ndarray, y: np.ndarray,
                                        validation_size=0.2, title_prefix=None):
    validation_cv = StratifiedShuffleSplit(n_splits=1, test_size=validation_size, random_state=0)
    train_ind, validation_ind = validation_cv.split(x, y).__next__()

    x_validation, y_validation = x[validation_ind], y[validation_ind]
    x, y = x[train_ind], y[train_ind]

    fig = plt.figure()
    ax = fig.add_subplot()
    title = 'Precision-Recall curves'
    if title_prefix is not None:
        title = title_prefix + ": " + title
    ax.set_title(title)

    for name, clf in classifiers.items():
        clf.fit(x, y)
        plot_precision_recall_curve(clf, x_validation, y_validation, name=name, ax=ax)

    ax.legend()
    plt.show()


def cross_validated_pr_curve(clf, x: np.ndarray, y: np.ndarray,
                             folds: int = 5, title_prefix=None):
    kfold = StratifiedKFold(n_splits=folds)

    all_p = []
    all_r = []
    all_thres = []

    for train_ind, test_ind in kfold.split(x, y):
        x_train, x_test = x[train_ind], x[test_ind]
        y_train, y_test = y[train_ind], y[test_ind]
        clf.fit(x, y)
        predictions = clf.predict_proba(x_test)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_test, predictions)
        all_p.append(precision)
        all_r.append(recall)
        all_thres.append(thresholds)
