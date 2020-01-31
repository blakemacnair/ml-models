from sklearn.svm import SVC
from mushrooms.mushroom_dataset import import_mushrooms_numpy
from titanic.titanic_dataset import import_cleaned_titanic_data
from sklearn.model_selection import ShuffleSplit


if __name__ == "__main__":
    cv = ShuffleSplit(n_splits=5, test_size=0.6, random_state=0)

    clf = SVC(kernel='linear')
    X, y = import_mushrooms_numpy(filepath="mushrooms/mushrooms.csv")

    print("Mushrooms")
    for train_ind, test_ind in cv.split(X, y):
        X_train, y_train = X[train_ind], y[train_ind]
        X_test, y_test = X[test_ind], y[test_ind]

        clf.fit(X_train, y_train)
        print(clf.score(X_test, y_test))

    clf_linear = SVC(kernel='linear')
    clf_rbf = SVC(kernel='rbf')
    clf_poly = SVC(kernel='poly')
    clf_sigmoid = SVC(kernel='sigmoid')

    scores = {
        'linear': [],
        'rbf': [],
        'poly': [],
        'sigmoid': []
    }

    X, y, X_test, test_ids = import_cleaned_titanic_data(directorypath="titanic/")

    for train_ind, test_ind in cv.split(X, y):
        X_train, y_train = X[train_ind], y[train_ind]
        X_test, y_test = X[test_ind], y[test_ind]

        clf_linear.fit(X_train, y_train)
        clf_poly.fit(X_train, y_train)
        clf_rbf.fit(X_train, y_train)
        clf_sigmoid.fit(X_train, y_train)

        scores['linear'].append(clf_linear.score(X_train, y_train))
        scores['poly'].append(clf_poly.score(X_train, y_train))
        scores['rbf'].append(clf_rbf.score(X_train, y_train))
        scores['sigmoid'].append(clf_sigmoid.score(X_train, y_train))

    for pair in scores.items():
        print(pair)

    print("Booty")
