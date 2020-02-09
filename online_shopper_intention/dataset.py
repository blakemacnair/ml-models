import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, OneHotEncoder


def load_shopper_intention_data(filepath='../data/online_shoppers_intention.csv'):
    data = pd.read_csv(filepath)

    categorical_cols = [
        'Month',
        'OperatingSystems',
        'Browser',
        'Region',
        'TrafficType',
        'VisitorType'
    ]
    cat_data = data[categorical_cols].copy()

    enc = OneHotEncoder(drop='first')
    enc = enc.fit(cat_data)

    cat_enc = enc.transform(cat_data).toarray()
    cat_data_enc = pd.DataFrame(cat_enc, columns=enc.get_feature_names(categorical_cols), index=data.index)
    data = data.join(cat_data_enc)
    data = data.drop(columns=categorical_cols)

    bool_cols = [
        'Weekend',
        'Revenue'
    ]
    for col in bool_cols:
        data[col] = data[col].map(lambda x: 1 if x else 0)

    data = data.fillna(data.median())
    return data


# TODO: Fix up for shopper intention data
def load_normalized_shopper_intention_data(filepath='../data/online_shoppers_intention.csv'):
    data = load_shopper_intention_data(filepath)
    return normalize_shopper_intention_data(data)


def load_shopper_intention_numpy(filepath='../data/online_shoppers_intention.csv'):
    data = load_shopper_intention_data(filepath)
    x = data.drop(columns='Revenue').to_numpy()
    y = data['Revenue'].to_numpy()
    return x, y


def load_normalized_shopper_intention_numpy(filepath='../data/online_shoppers_intention.csv'):
    data = load_normalized_shopper_intention_data(filepath)
    x = data.drop(columns=['Revenue']).to_numpy()
    y = data['Revenue'].to_numpy()
    return x, y


def normalize_shopper_intention_data(data: pd.DataFrame):
    v_data = data[data.columns[:10]]
    v = v_data.to_numpy()
    v_scaled = scale(v)
    v_data_scaled = pd.DataFrame(v_scaled, index=v_data.index, columns=v_data.columns)

    data_scaled = data.copy()
    data_scaled.update(v_data_scaled)
    return data_scaled


if __name__ == "__main__":
    data = load_shopper_intention_data()

    # Demonstrate that the data might be fairly difficult to separate, at least without expanding the data into
    # higher dimensionality
    g = sb.pairplot(data, hue='Revenue', vars=np.concatenate((data.columns[:2], data.columns[4:8])))
    g.fig.show()

    print("Dope.")
