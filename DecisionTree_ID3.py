import sklearn.datasets as skdata
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


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


def find_best_attribute_to_split(s: pd.DataFrame, attributes: pd.Index, target_attribute: str):
    gains = []
    for a in attributes:
        gains.append(information_gain(s, a, target_attribute))

    return attributes[gains.index(max(gains))], max(gains)


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
