import sklearn.datasets as skdata
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


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


class ID3Node:
    def __init__(self, s: pd.DataFrame, attributes: pd.Index, target_attribute: str,
                 depth: int = 1, max_depth: int = np.inf, min_gain=0.4):
        self.prediction: Optional[Any] = None
        self.children: Optional[Dict[Any, ID3Node]] = None
        self.split_attribute: Optional[str] = None

        if len(attributes) <= 1 or depth >= max_depth:
            targets = s[target_attribute]
            self.prediction = targets.value_counts().index[0]
        else:
            self.split_attribute, gain = find_best_attribute_to_split(s, attributes, target_attribute)
            if gain < min_gain:
                targets = s[target_attribute]
                self.prediction = targets.value_counts().index[0]
                return

            unique_values = s[self.split_attribute].unique()

            next_attributes = attributes.drop([self.split_attribute])
            self.children = {value: ID3Node(s[s[self.split_attribute] == value],
                                            next_attributes,
                                            target_attribute,
                                            depth=depth + 1,
                                            max_depth=max_depth,
                                            min_gain=min_gain)
                             for value in unique_values}

    def size(self):
        if self.children is None:
            return 1
        return 1 + sum([c.size() for c in self.children.values()])

    def depth(self):
        if self.children is None:
            return 1
        return 1 + max([c.depth() for c in self.children.values()])

    def predict(self, samples: pd.DataFrame):
        if self.prediction is not None:
            return pd.Series(self.prediction, index=samples.index)
        else:
            predictions = pd.Series(None, index=samples.index)
            for (key, child) in self.children.items():
                subset = samples[samples[self.split_attribute] == key]
                predictions = predictions.combine_first(child.predict(subset))
            return predictions


if __name__ == "__main__":
    set = skdata.make_classification()
