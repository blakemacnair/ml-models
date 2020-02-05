import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.tree import DecisionTreeClassifier

from titanic.titanic_dataset import import_cleaned_titanic_data

from metrics import plot_compare_precision_recall_curve, plot_compare_learning_curve, plot_compare_roc_curve

if __name__ == "__main__":
    cv = ShuffleSplit(n_splits=10, test_size=0.5, random_state=0)

    x, y, x_test, passenger_ids = import_cleaned_titanic_data()

    best = None
    best_acc = 0

    for train_ind, test_ind in cv.split(x, y):
        X_train, X_test = x[train_ind], x[test_ind]
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

    y_test = best.predict(x_test)
    df_pred = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived'   : y_test.astype(int)
    })
    df_pred.to_csv("pred.csv", index=False)
    target = 'kaggle competitions submit -c titanic -f submission.csv -m "Decision Tree Post-Cleaning"'

    # TODO: Analyze these plots, and try to print them in more resolution!
    # plt.figure()
    # tree.plot_tree(best, filled=True)
    # plt.show()

    # models = {
    #     'impurity 2e-2': DecisionTreeClassifier(min_impurity_decrease=2e-2),
    #     'impurity 2e-3': DecisionTreeClassifier(min_impurity_decrease=2e-3),  # The best performing impurity
    #     'impurity 2e-5': DecisionTreeClassifier(min_impurity_decrease=2e-5),
    #     'impurity 2e-7': DecisionTreeClassifier(min_impurity_decrease=2e-7)
    # }

    # models = {
    #     'min samples 1': DecisionTreeClassifier(min_samples_leaf=1, min_impurity_decrease=2e-3),
    #     'min samples 3': DecisionTreeClassifier(min_samples_leaf=3, min_impurity_decrease=2e-3),
    #     'min samples 5': DecisionTreeClassifier(min_samples_leaf=5, min_impurity_decrease=2e-3),
    #     'min samples 9': DecisionTreeClassifier(min_samples_leaf=9, min_impurity_decrease=2e-3),  # best min samples
    # }

    models = {
        'GINI': DecisionTreeClassifier(criterion='gini', min_samples_leaf=9, min_impurity_decrease=2e-3),
        'entropy': DecisionTreeClassifier(criterion='entropy', min_samples_leaf=9, min_impurity_decrease=2e-3)
    }

    cv = ShuffleSplit(n_splits=10, test_size=0.2)
    train_sizes = np.linspace(0.3, 1.0, 100)

    plot_compare_roc_curve(models, x, y)
    plot_compare_precision_recall_curve(models, x, y)
    plot_compare_learning_curve(models, x, y, cv=cv, train_sizes=train_sizes)
