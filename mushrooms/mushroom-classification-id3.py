import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit
from DecisionTree_ID3 import ID3Node


def import_mushrooms() -> pd.DataFrame:
    return pd.read_csv("mushrooms.csv")


if __name__ == "__main__":
    shrooms = import_mushrooms()
    target = 'class'
    targets = shrooms[target]

    attributes = shrooms.keys().drop([target])

    cv = ShuffleSplit(n_splits=3, test_size=0.7, random_state=0)
    splits = cv.split(shrooms, targets)
    for train_index, test_index in cv.split(shrooms, targets):
        train_index = sorted(train_index)
        test_index = sorted(test_index)

        train_x = shrooms.iloc[train_index]
        train_y = targets.iloc[train_index]

        test_x = shrooms.iloc[test_index]
        test_y = targets.iloc[test_index]

        id3 = ID3Node(train_x, attributes, target, min_gain=0.1)
        pred = id3.predict(test_x)
        acc = np.sum(pred == test_y) / len(pred)
        print("Accuracy: {}\nDepth: {}\nSize: {}\n".format(acc, id3.depth(), id3.size()))
