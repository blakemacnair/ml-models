import numpy as np

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import get_scorer, make_scorer, f1_score, roc_auc_score
from sklearn.feature_selection import RFECV

from titanic.titanic_dataset import import_cleaned_titanic_data
from credit_card_fraud.dataset import load_normalized_credit_fraud_numpy, load_credit_fraud_numpy

from metrics import plot_compare_precision_recall_curve, plot_compare_learning_curve, plot_compare_roc_curve

if __name__ == "__main__":
    # x, y, _, _ = import_cleaned_titanic_data(directorypath='titanic/')
    x, y = load_credit_fraud_numpy(filepath='data/creditcard.csv')

    split_cv = StratifiedKFold(n_splits=7, shuffle=True, random_state=0)
    train_ind, test_ind = list(split_cv.split(x, y))[0]

    x_test, y_test = x[test_ind], y[test_ind]
    x_train, y_train = x[train_ind], y[train_ind]

    # models = {
    #     'Adaboost': AdaBoostClassifier(),
    #     'Gradient': GradientBoostingClassifier()
    # }

    # plot_compare_roc_curve(models, x, y)
    # plot_compare_precision_recall_curve(models, x, y)
    # plot_compare_learning_curve(models, x, y)

    clf = AdaBoostClassifier(random_state=0)

    y_true, y_pred = y_test, clf.predict(x_test)
    print(classification_report(y_true, y_pred))
    print()
    print("Booty")
