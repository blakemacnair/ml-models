import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from DecisionTree_ID3 import clean_na_values
import re

from sklearn.neighbors import KNeighborsClassifier


def import_titanic_train_test() -> (pd.DataFrame, pd.DataFrame):
    return pd.read_csv("train.csv"), pd.read_csv("test.csv")


def clean_titanic(s: pd.DataFrame, k_age=None, k_fare=None, mc_names=None, mc_tickets=None, enc=None):
    s = s.copy()
    clean_cabin_numbers(s)
    k_age = clean_age(s, k_age)
    k_fare = clean_fares(s, k_fare)
    mc_names = clean_names(s, mc_names)
    mc_tickets = clean_tickets(s, mc_tickets)
    clean_na_values(s)
    s = s.drop(columns=["Age", "Cabin", "Fare", "Name", "Ticket"])
    enc, old_columns = encode_non_numeric_columns(s, enc)
    s = s.drop(columns=old_columns)

    return s, k_age, k_fare, mc_names, mc_tickets, enc


def clean_cabin_numbers(s: pd.DataFrame):
    s["Cabin"].loc[pd.isna(s["Cabin"])] = 'Z000'
    cabins = s["Cabin"].copy()
    letters = cabins \
        .apply(lambda x: re.findall('[^0-9]+', x)[0]) \
        .apply(lambda x: x[0] if len(x) > 0 else 'Z')
    numbers = cabins \
        .apply(lambda x: re.findall('[0-9]+', x)) \
        .apply(lambda x: x[0] if len(x) > 0 else '-1') \
        .apply(lambda x: int(x))
    s["CabinLetter"] = letters
    s["CabinNumber"] = numbers


def clean_age(s: pd.DataFrame, k: KNeighborsClassifier = None):
    ages = s["Age"].copy()
    mean_age = np.round(np.mean(ages))
    ages[pd.isna(ages)] = mean_age
    s["Age"] = ages

    if k is None:
        k = KNeighborsClassifier()
        k.fit(ages.array.to_numpy().reshape(-1, 1), s["Survived"].to_numpy())

    p = k.predict(ages.array.to_numpy().reshape(-1, 1))
    age_survival = pd.Series(p, index=ages.index)
    s["AgeSurvival"] = age_survival

    return k


def clean_fares(s: pd.DataFrame, k: KNeighborsClassifier = None):
    fares = s["Fare"].copy()
    mean_age = np.round(np.mean(fares))
    fares[pd.isna(fares)] = mean_age
    s["Fare"] = fares

    if k is None:
        k = KNeighborsClassifier()
        k.fit(fares.array.to_numpy().reshape(-1, 1), s["Survived"].to_numpy())

    p = k.predict(fares.array.to_numpy().reshape(-1, 1))
    fare_survival = pd.Series(p, index=fares.index)
    s["FareSurvival"] = fare_survival

    return k


def clean_names(s: pd.DataFrame, most_common=None):
    if most_common is None:
        n = s["Name"].copy()
        n1 = n.apply(lambda x: np.array(x.split(' ')))
        name_comps = np.concatenate(n1)
        name_series = pd.Series(name_comps)
        most_common = name_series.value_counts()[:10]

    for c in most_common.index:
        s[c] = s["Name"].apply(lambda x: 1 if c in x else 0)

    return most_common


def clean_tickets(s: pd.DataFrame, most_common=None):
    if most_common is None:
        t = s["Ticket"].copy()
        letters = t.apply(lambda x: re.findall('[^0-9./ ]+', x))
        t_comps = np.concatenate(letters)
        t_series = pd.Series(t_comps)
        most_common = t_series.value_counts()[:10]

    for c in most_common.index:
        s[c] = s["Ticket"].apply(lambda x: 1 if c in x else 0)

    return most_common


def encode_non_numeric_columns(s: pd.DataFrame, enc=None):
    cols = ["PassengerId"]
    if "Survived" in s.columns:
        cols.append("Survived")
    sa = s.copy().drop(columns=cols)
    numeric = []
    for a in sa.columns:
        if np.issubdtype(sa[a].dtype, np.number):
            numeric.append(a)
    sa = sa.drop(columns=numeric)
    cols = sa.columns

    if enc is None:
        enc = OneHotEncoder(handle_unknown='ignore')
        enc = enc.fit(sa)

    sa_enc = enc.transform(sa).toarray()
    sa = pd.DataFrame(sa_enc)
    s.join(sa)

    return enc, cols


if __name__ == "__main__":
    train, test = import_titanic_train_test()
    s, k_age, k_fare, mc_names, mc_tickets, enc = clean_titanic(train)

    X = s.drop(columns=["PassengerId", "Survived"]).to_numpy()
    y = s["Survived"].to_numpy()

    cv = ShuffleSplit(n_splits=20, test_size=0.5, random_state=0)

    best = None
    best_acc = 0

    for train_ind, test_ind in cv.split(X, y):
        X_train, X_test = X[train_ind], X[test_ind]
        y_train, y_test = y[train_ind], y[test_ind]

        d = DecisionTreeClassifier(min_impurity_decrease=5e-3)
        d.fit(X_train, y_train)
        print(d.get_depth())

        train_score = d.score(X_train, y_train)
        test_score = d.score(X_test, y_test)
        print("Train acc: {}\nTest acc: {}\n--------".format(train_score, test_score))

        if test_score > best_acc:
            best_acc = test_score
            best = d

    s_test, _, _, _, _, _ = clean_titanic(test, k_age, k_fare, mc_names, mc_tickets, enc)
    X_test = s_test.drop(columns=["PassengerId"]).to_numpy()
    y = best.predict(X_test)

    df_pred = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived'   : y.astype(int)
    })

    df_pred.to_csv("pred.csv", index=False)

    target = 'kaggle competitions submit -c titanic -f submission.csv -m "Decision Tree Post-Cleaning"'
