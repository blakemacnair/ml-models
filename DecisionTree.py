from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.base import clone
import matplotlib.pylab as plt
from sklearn.model_selection import ShuffleSplit
from sklearn.tree import DecisionTreeClassifier

from titanic.titanic_dataset import import_cleaned_titanic_data
from mushrooms.mushroom_dataset import import_mushrooms_numpy
from metrics import plot_nodes_vs_alpha, plot_cost_complexity_pruning_path, plot_acc_alpha_train_vs_test

from metrics import plot_compare_precision_recall_curve, plot_compare_learning_curve, plot_compare_roc_curve


def clean_na_values(s: pd.DataFrame):
    means = s.mean()
    for c in s.columns:
        if c in means.keys():
            s[c].loc[s[c].isna()] = means[c]
        else:
            m = s[c].value_counts().index[0]
            s[c].loc[s[c].isna()] = m


if __name__ == "__main__":
    x, y, _, _ = import_cleaned_titanic_data(directorypath="titanic/")

    models_min_impurity_decrease = {
        'impurity 2e-2': DecisionTreeClassifier(min_impurity_decrease=2e-2),
        'impurity 2e-3': DecisionTreeClassifier(min_impurity_decrease=2e-3),  # The best performing impurity
        'impurity 2e-5': DecisionTreeClassifier(min_impurity_decrease=2e-5),
        'impurity 2e-7': DecisionTreeClassifier(min_impurity_decrease=2e-7)
    }

    models_min_samples_leaf = {
        'min samples 1': DecisionTreeClassifier(min_samples_leaf=1, min_impurity_decrease=2e-3),
        'min samples 3': DecisionTreeClassifier(min_samples_leaf=3, min_impurity_decrease=2e-3),
        'min samples 5': DecisionTreeClassifier(min_samples_leaf=5, min_impurity_decrease=2e-3),
        'min samples 9': DecisionTreeClassifier(min_samples_leaf=9, min_impurity_decrease=2e-3),  # best min samples
    }

    models_criterion = {
        'GINI'   : DecisionTreeClassifier(criterion='gini', min_samples_leaf=9, min_impurity_decrease=2e-3),
        'entropy': DecisionTreeClassifier(criterion='entropy', min_samples_leaf=9, min_impurity_decrease=2e-3)
    }

    models_pruning_alpha = {
        'a.0|Mins'   : DecisionTreeClassifier(min_samples_leaf=9, min_impurity_decrease=2e-3, ccp_alpha=0.0),
        'a.02|Mins'  : DecisionTreeClassifier(min_samples_leaf=9, min_impurity_decrease=2e-3, ccp_alpha=0.02),
        'a.0|NoMins' : DecisionTreeClassifier(min_impurity_decrease=2e-3, ccp_alpha=0.0),
        'a.02|NoMins': DecisionTreeClassifier(min_impurity_decrease=2e-3, ccp_alpha=0.02),
    }

    cv = ShuffleSplit(n_splits=20, test_size=0.5)
    train_sizes = np.linspace(0.3, 1.0, 100)

    plot_compare_roc_curve(models_pruning_alpha, x, y)
    plot_compare_precision_recall_curve(models_pruning_alpha, x, y)
    plot_compare_learning_curve(models_pruning_alpha, x, y, cv=cv, train_sizes=train_sizes)

    clf = DecisionTreeClassifier(criterion="entropy", random_state=0)

    plot_cost_complexity_pruning_path(clf, x, y)
    plot_nodes_vs_alpha(clf, x, y)

    cv_validation = ShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
    train, test = list(cv_validation.split(x, y))[0]
    x_train, y_train = x[train], y[train]
    x_test, y_test = x[test], y[test]

    clf = DecisionTreeClassifier(criterion="entropy", random_state=0)
    plot_acc_alpha_train_vs_test(clf, x_train, y_train, x_test, y_test)

    clf = DecisionTreeClassifier(criterion="gini", splitter="best", random_state=0)
    plot_acc_alpha_train_vs_test(clf, x_train, y_train, x_test, y_test)

    clf = DecisionTreeClassifier(criterion="gini", splitter="random", random_state=0)
    plot_acc_alpha_train_vs_test(clf, x_train, y_train, x_test, y_test)
