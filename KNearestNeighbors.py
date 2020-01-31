from sklearn.neighbors import KNeighborsClassifier
from mushrooms.mushroom_dataset import import_mushrooms_numpy
from titanic.titanic_dataset import import_cleaned_titanic_data
from sklearn.model_selection import ShuffleSplit


if __name__ == "__main__":
    cv = ShuffleSplit(n_splits=5, test_size=0.6, random_state=0)

    clf = KNeighborsClassifier()
    X, y = import_mushrooms_numpy(filepath="mushrooms/mushrooms.csv")

    # print("Mushrooms")
    # for train_ind, test_ind in cv.split(X, y):
    #     X_train, y_train = X[train_ind], y[train_ind]
    #     X_test, y_test = X[test_ind], y[test_ind]
    #
    #     clf.fit(X_train, y_train)
    #     print(clf.score(X_test, y_test))

    # Algorithm comparison
    clf_auto = KNeighborsClassifier(algorithm='auto', weights='distance')
    clf_ball_tree = KNeighborsClassifier(algorithm='ball_tree', weights='distance')
    clf_kd_tree = KNeighborsClassifier(algorithm='kd_tree', weights='distance')
    clf_brute = KNeighborsClassifier(algorithm='brute', weights='distance')

    scores = {
        'auto': [],
        'ball_tree': [],
        'kd_tree': [],
        'brute': []
    }

    X, y, X_test, test_ids = import_cleaned_titanic_data(directorypath="titanic/")

    for train_ind, test_ind in cv.split(X, y):
        X_train, y_train = X[train_ind], y[train_ind]
        X_test, y_test = X[test_ind], y[test_ind]

        clf_auto.fit(X_train, y_train)
        clf_ball_tree.fit(X_train, y_train)
        clf_kd_tree.fit(X_train, y_train)
        clf_brute.fit(X_train, y_train)

        scores['auto'].append(clf_auto.score(X_train, y_train))
        scores['ball_tree'].append(clf_ball_tree.score(X_train, y_train))
        scores['kd_tree'].append(clf_kd_tree.score(X_train, y_train))
        scores['brute'].append(clf_brute.score(X_train, y_train))

    for pair in scores.items():
        print(pair)

    print("Booty")
