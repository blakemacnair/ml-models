import numpy as np

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

from titanic.titanic_dataset import import_cleaned_titanic_data
from online_shopper_intention.dataset import load_shopper_intention_numpy
from credit_card_fraud.dataset import load_credit_fraud_numpy

from metrics import plot_compare_precision_recall_curve, plot_compare_learning_curve, plot_compare_roc_curve, \
    plot_roc_crossval

if __name__ == "__main__":
    models = {
        'Adaboost': AdaBoostClassifier(),
        'Gradient': GradientBoostingClassifier()
    }

    x, y = load_credit_fraud_numpy(filepath='data/creditcard.csv')

    split_cv = StratifiedKFold(n_splits=7, shuffle=True, random_state=0)
    train_ind, test_ind = list(split_cv.split(x, y))[0]

    x_test, y_test = x[test_ind], y[test_ind]
    x_train, y_train = x[train_ind], y[train_ind]

    # plot_compare_roc_curve(models, x, y)
    # plot_compare_precision_recall_curve(models, x, y)
    # plot_compare_learning_curve(models, x, y)

    x, y = load_shopper_intention_numpy(filepath='data/online_shoppers_intention.csv')

    plot_compare_roc_curve(models, x, y)
    plot_compare_precision_recall_curve(models, x, y)
    # plot_compare_learning_curve(models, x, y)
    plot_roc_crossval(models['Gradient'], x, y)

    print("Booty")
