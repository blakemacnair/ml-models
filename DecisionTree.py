import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import ShuffleSplit
from sklearn.tree import DecisionTreeClassifier

from titanic.titanic_dataset import import_cleaned_titanic_data
from credit_card_fraud.dataset import load_credit_fraud_numpy

from metrics import plot_nodes_vs_alpha, plot_cost_complexity_pruning_path, plot_acc_alpha_train_vs_test
from metrics import plot_compare_precision_recall_curve, plot_compare_learning_curve, plot_compare_roc_curve

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, get_scorer, make_scorer, accuracy_score, precision_score


if __name__ == "__main__":
    # x, y, _, _ = import_cleaned_titanic_data(directorypath="titanic/")
    x, y = load_credit_fraud_numpy(filepath="data/creditcard.csv")
    # x = x[:, 1:19]

    split_cv = ShuffleSplit(n_splits=1, test_size=0.7)
    train_inds, validation_inds = list(split_cv.split(x, y))[0]

    x_test, y_test = x[validation_inds], y[validation_inds]
    x, y = x[train_inds], y[train_inds]

    models_min_impurity_decrease = {
        'impurity 2e-2': DecisionTreeClassifier(min_impurity_decrease=2e-2, class_weight='balanced'),
        'impurity 2e-3': DecisionTreeClassifier(min_impurity_decrease=2e-3, class_weight='balanced'),
        'impurity 2e-5': DecisionTreeClassifier(min_impurity_decrease=2e-5, class_weight='balanced'),
        'impurity 2e-7': DecisionTreeClassifier(min_impurity_decrease=2e-7, class_weight='balanced')
    }
    target_imp_dec = 2e-5

    models_min_samples_leaf = {
        'min samples 1': DecisionTreeClassifier(min_samples_leaf=1, min_impurity_decrease=target_imp_dec,
                                                class_weight='balanced'),
        'min samples 3': DecisionTreeClassifier(min_samples_leaf=3, min_impurity_decrease=target_imp_dec,
                                                class_weight='balanced'),
        'min samples 5': DecisionTreeClassifier(min_samples_leaf=5, min_impurity_decrease=target_imp_dec,
                                                class_weight='balanced'),
        'min samples 9': DecisionTreeClassifier(min_samples_leaf=9, min_impurity_decrease=target_imp_dec,
                                                class_weight='balanced'),
    }
    target_min_samples = 1

    models_criterion = {
        'gini'   : DecisionTreeClassifier(criterion='gini', min_samples_leaf=target_min_samples,
                                          min_impurity_decrease=target_imp_dec, class_weight='balanced'),
        'entropy': DecisionTreeClassifier(criterion='entropy', min_samples_leaf=target_min_samples,
                                          min_impurity_decrease=target_imp_dec, class_weight='balanced')
    }
    c = 'entropy'

    models_pruning_alpha = {
        'a.0'  : DecisionTreeClassifier(ccp_alpha=0.0, class_weight='balanced'),
        'a.001': DecisionTreeClassifier(ccp_alpha=0.001, class_weight='balanced'),
        'a.002': DecisionTreeClassifier(ccp_alpha=0.002, class_weight='balanced'),
        'a.003': DecisionTreeClassifier(ccp_alpha=0.003, class_weight='balanced')
    }

    models_balancing = {
        'balanced'  : DecisionTreeClassifier(criterion=c, class_weight='balanced'),
        'unbalanced': DecisionTreeClassifier(criterion=c)
    }

    # cv = ShuffleSplit(n_splits=10, test_size=0.6, random_state=0)
    # train_sizes = np.linspace(0.2, 1, 7)
    # cmp_models = models_balancing

    # plot_compare_roc_curve(cmp_models, x, y)
    # plot_compare_precision_recall_curve(cmp_models, x, y)
    # plot_compare_learning_curve(cmp_models, x, y, cv=cv, train_sizes=train_sizes, scoring='precision')

    clf_credit = DecisionTreeClassifier(criterion='entropy', class_weight=None, random_state=0)

    scoring = {
        'AUC'      : 'roc_auc',
        'Precision': make_scorer(precision_score)
    }

    param_grid = {
        'min_samples_split': np.linspace(2, 150, 5, dtype=np.int),
        'ccp_alpha': np.linspace(0, 1e-3, 5)
    }

    gs = GridSearchCV(clf_credit,
                      param_grid=param_grid,
                      scoring=scoring, refit='Precision', return_train_score=True)

    gs.fit(x, y)
    print("Precision scores on development set:")
    print()
    means = gs.cv_results_['mean_test_Precision']
    stds = gs.cv_results_['std_test_Precision']
    for mean, std, params in zip(means, stds, gs.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Best parameters set found on development set:")
    print()
    print(gs.best_params_)
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, gs.predict(x_test)
    print(classification_report(y_true, y_pred))
    print()
    # Best: {'ccp_alpha': 0.0005555555555555556, 'class_weight': None, 'criterion': 'entropy',
    # 'min_impurity_decrease': 0.0, 'min_samples_split': 62}

    # clf = DecisionTreeClassifier(criterion="entropy", random_state=0, class_weight='balanced')
    #
    # plot_cost_complexity_pruning_path(clf, x, y)
    # plot_nodes_vs_alpha(clf, x, y)
    #
    # cv_validation = ShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
    # train, test = list(cv_validation.split(x, y))[0]
    # x_train, y_train = x[train], y[train]
    # x_test, y_test = x[test], y[test]
    #
    # clf = DecisionTreeClassifier(criterion="entropy", random_state=0, class_weight='balanced')
    # plot_acc_alpha_train_vs_test(clf, x_train, y_train, x_test, y_test)
