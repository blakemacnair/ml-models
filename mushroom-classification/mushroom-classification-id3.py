from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from DecisionTree import Node


def import_mushrooms() -> pd.DataFrame:
    return pd.read_csv("mushrooms.csv")


if __name__ == "__main__":
    shrooms = import_mushrooms()
    target = 'class'
    targets = shrooms[target]

    attributes = shrooms.keys().drop([target])
    node = Node(shrooms, attributes, target)
    predictions = node.predict(shrooms)
    is_correct = targets == predictions
    acc = np.sum(is_correct) / len(predictions)
    print("Accuracy: {}".format(acc))

    cv = ShuffleSplit(n_splits=7, test_size=0.4, random_state=0)
    splits = cv.split(shrooms, targets)
    for train_index, test_index in cv.split(shrooms, targets):
        train_index = sorted(train_index)
        test_index = sorted(test_index)

        train_x = shrooms.iloc[train_index]
        train_y = targets.iloc[train_index]

        test_x = shrooms.iloc[test_index]
        test_y = targets.iloc[test_index]

        id3 = Node(train_x, attributes, target, min_gain=0.4)
        pred = id3.predict(test_x)
        acc = np.sum(pred == test_y) / len(pred)
        print(acc)
