import pandas as pd
import numpy as np
import seaborn as sb
from sklearn import preprocessing


def load_credit_fraud_data(filepath='../data/creditcard.csv'):
    return pd.read_csv(filepath)


def load_normalized_credit_fraud_data(filepath='../data/creditcard.csv'):
    data = load_credit_fraud_data(filepath)
    return normalize_credit_fraud_data(data)


def load_credit_fraud_numpy(filepath='../data/creditcard.csv'):
    data = load_credit_fraud_data(filepath)
    xy = data.to_numpy()
    x = xy[:, :-1]
    y = xy[:, -1]
    return x, y


def load_normalized_credit_fraud_numpy(filepath='../data/creditcard.csv'):
    data = load_normalized_credit_fraud_data(filepath)
    xy = data.to_numpy()
    x = xy[:, :-1]
    y = xy[:, -1]
    return x, y


def normalize_credit_fraud_data(data: pd.DataFrame):
    v_data = data.drop(columns=['Class', 'Amount', 'Time'])
    v = v_data.to_numpy()
    v_scaled = preprocessing.scale(v)
    v_data_scaled = pd.DataFrame(v_scaled, index=v_data.index, columns=v_data.columns)

    data_scaled = data.copy()
    data_scaled.update(v_data_scaled)
    return data_scaled


if __name__ == "__main__":
    data = load_credit_fraud_data()
    data_np = data.to_numpy()

    # Analyze the class breakdown: 284,315 non-fraud (0), 492 fraud (1)
    class_breakdown = pd.value_counts(data_np[:, -1])
    total = np.sum(class_breakdown)
    print("Class breakdown: \n"
          "Non-fraudulent: {}%\t({})\n"
          "    Fraudulent: {}% \t({})".format(np.round(100 * class_breakdown[0] / total, decimals=3),
                                              class_breakdown[0],
                                              np.round(100 * class_breakdown[1] / total, decimals=3),
                                              class_breakdown[1]))

    # TODO: Consider weighting the classes to counteract the high imbalance
    # https://scikit-learn.org/stable/modules/svm.html#unbalanced-problems

    # Normalize the V_ data columns, leave Amount and Time as-is
    data_scaled = normalize_credit_fraud_data(data)

    # TODO: Consider PCA to reduce dimensionality and complexity
    # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html

    # Show separability of some sample columns in dataset
    sample = data_scaled.iloc[:20000]
    g = sb.pairplot(sample[['V1', 'V2', 'V3', 'V4', 'V5', 'Class']], hue='Class')
    g.fig.show()

    # TODO: Consider polynomial combinations of features into higher dimensions
    # https://scikit-learn.org/stable/modules/preprocessing.html#generating-polynomial-features

    print("Done")
