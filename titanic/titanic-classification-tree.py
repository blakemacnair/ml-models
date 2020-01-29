import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from titanic.titanic_dataset import import_titanic_train_test, clean_titanic

if __name__ == "__main__":
    train, test = import_titanic_train_test()
    s, k_age, k_fare, mc_names, mc_tickets, enc = clean_titanic(train)

    X = s.drop(columns=["PassengerId", "Survived"]).to_numpy()
    y = s["Survived"].to_numpy()

    cv = ShuffleSplit(n_splits=10, test_size=0.5, random_state=0)

    best = None
    best_acc = 0

    for train_ind, test_ind in cv.split(X, y):
        X_train, X_test = X[train_ind], X[test_ind]
        y_train, y_test = y[train_ind], y[test_ind]

        d = DecisionTreeClassifier(min_impurity_decrease=2e-3)
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
    y_test = best.predict(X_test)

    # TODO: Analyze these plots, and try to print them in more resolution!
    plt.figure()
    tree.plot_tree(best, filled=True)
    plt.show()

    df_pred = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived'   : y_test.astype(int)
    })

    df_pred.to_csv("pred.csv", index=False)

    target = 'kaggle competitions submit -c titanic -f submission.csv -m "Decision Tree Post-Cleaning"'
