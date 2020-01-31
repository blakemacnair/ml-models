from sklearn.preprocessing import OneHotEncoder
from torch import Tensor
import pandas as pd
import numpy as np


def mushrooms_to_numeric(s: pd.DataFrame):
    enc = OneHotEncoder(drop='first')
    enc = enc.fit(s)

    s_enc = enc.transform(s).toarray()
    return s_enc, enc


def import_mushrooms(filepath="mushrooms.csv") -> pd.DataFrame:
    return pd.read_csv(filepath)


def import_mushrooms_numpy(filepath="mushrooms.csv"):
    s = import_mushrooms(filepath)

    s_enc, enc = mushrooms_to_numeric(s)
    X = s_enc[:, 1:]
    y = s_enc[:, 0]
    return X, y


def import_mushrooms_pytorch(filepath="mushrooms.csv"):
    s = import_mushrooms(filepath)

    s_enc, enc = mushrooms_to_numeric(s)
    s_enc = Tensor(s_enc)
    X = s_enc[:, 1:]
    y = s_enc[:, 0].reshape(-1, 1)
    y = Tensor(np.concatenate((y, 1 - y), axis=1))
    return X, y
