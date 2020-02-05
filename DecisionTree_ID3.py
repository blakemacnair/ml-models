from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.tree import DecisionTreeClassifier

from titanic.titanic_dataset import import_cleaned_titanic_data

from metrics import plot_compare_precision_recall_curve, plot_compare_learning_curve, plot_compare_roc_curve



def entropy(s: pd.DataFrame, a: str):
    a_values = s[a]
    n = len(a_values)
    values = a_values.unique()
    p_values = [len(a_values[a_values == value]) / n for value in values]

    return - np.sum([p * np.log2(p) for p in p_values])


def information_gain(s: pd.DataFrame, a: str, target_attribute: str):
    e_total = entropy(s, target_attribute)

    n = len(s)
    a_values = s[a]
    values = a_values.unique()
    e_given_a = np.sum([entropy(s[s[a] == value], target_attribute) * len(s[s[a] == value]) / n for value in values])

    return e_total - e_given_a


def split_information(s: pd.DataFrame, a: str):
    s_a = s[a]
    uniques = [c for c in s_a.unique() if c is not np.nan]
    size_a = s_a.size
    return -sum([(s_a[s_a == c].size / size_a) * np.log2(s_a[s_a == c].size / size_a) for c in uniques
                 if s_a[s_a == c].size != 0])


def gain_ratio(s: pd.DataFrame, a: str, target_attribute: str):
    return information_gain(s, a, target_attribute) / split_information(s, a)


def find_best_attribute_to_split(s: pd.DataFrame, attributes: pd.Index, target_attribute: str):
    gains = []
    for a in attributes:
        if is_attribute_numeric(s, a):
            gains.append(numeric_information_gain(s, a, target_attribute))
        else:
            gains.append(gain_ratio(s, a, target_attribute))

    return attributes[gains.index(max(gains))], max(gains)


def numeric_information_gain(s: pd.DataFrame, num_attribute: str, target_attribute: str):
    k = KNeighborsClassifier()
    x = s[num_attribute].to_numpy().reshape(-1, 1)
    y = s[target_attribute].to_numpy()
    # TODO: Need to be able to factorize the target attribute column for these tasks
    k.fit(x, y)
    p = k.predict(x)
    p_series = pd.Series(p, index=s.index)
    cp = pd.DataFrame(data=s[target_attribute].copy())
    cp[num_attribute] = p_series
    ig = information_gain(cp, num_attribute, target_attribute)
    return ig, k


def is_attribute_numeric(s: pd.DataFrame, a: str, min_classes=10):
    if np.issubdtype(s[a].dtype, np.number) and s[a].unique().size > min_classes:
        return True
    else:
        return False


def clean_na_values(s: pd.DataFrame):
    means = s.mean()
    for c in s.columns:
        if c in means.keys():
            s[c].loc[s[c].isna()] = means[c]
        else:
            m = s[c].value_counts().index[0]
            s[c].loc[s[c].isna()] = m


if __name__ == "__main__":
    cv = ShuffleSplit(n_splits=10, test_size=0.5, random_state=0)

    x, y, x_test, passenger_ids = import_cleaned_titanic_data(directorypath="titanic/")

    # models = {
    #     'impurity 2e-2': DecisionTreeClassifier(min_impurity_decrease=2e-2),
    #     'impurity 2e-3': DecisionTreeClassifier(min_impurity_decrease=2e-3),  # The best performing impurity
    #     'impurity 2e-5': DecisionTreeClassifier(min_impurity_decrease=2e-5),
    #     'impurity 2e-7': DecisionTreeClassifier(min_impurity_decrease=2e-7)
    # }

    # models = {
    #     'min samples 1': DecisionTreeClassifier(min_samples_leaf=1, min_impurity_decrease=2e-3),
    #     'min samples 3': DecisionTreeClassifier(min_samples_leaf=3, min_impurity_decrease=2e-3),
    #     'min samples 5': DecisionTreeClassifier(min_samples_leaf=5, min_impurity_decrease=2e-3),
    #     'min samples 9': DecisionTreeClassifier(min_samples_leaf=9, min_impurity_decrease=2e-3),  # best min samples
    # }

    models = {
        'GINI'   : DecisionTreeClassifier(criterion='gini', min_samples_leaf=9, min_impurity_decrease=2e-3),
        'entropy': DecisionTreeClassifier(criterion='entropy', min_samples_leaf=9, min_impurity_decrease=2e-3)
    }

    cv = ShuffleSplit(n_splits=10, test_size=0.2)
    train_sizes = np.linspace(0.3, 1.0, 100)

    plot_compare_roc_curve(models, x, y)
    plot_compare_precision_recall_curve(models, x, y)
    plot_compare_learning_curve(models, x, y, cv=cv, train_sizes=train_sizes)

