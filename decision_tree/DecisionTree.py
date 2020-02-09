import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier

from online_shopper_intention.dataset import load_shopper_intention_numpy
from credit_card_fraud.dataset import load_credit_fraud_numpy

from metrics import plot_nodes_vs_alpha, plot_cost_complexity_pruning_path, plot_acc_alpha_train_vs_test
from metrics import plot_compare_precision_recall_curve, plot_compare_learning_curve, plot_compare_roc_curve

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict, cross_validate, train_test_split
from sklearn.metrics import classification_report, get_scorer, make_scorer, accuracy_score, precision_score


def compare_models_all_metrics(models,
                               x,
                               y,
                               train_sizes,
                               scoring='balanced_accuracy',
                               cv=StratifiedShuffleSplit(n_splits=5, test_size=0.7),
                               title_prefix=None):
    plot_compare_roc_curve(models, x, y, title_prefix=title_prefix)
    plot_compare_precision_recall_curve(models, x, y, title_prefix=title_prefix)
    plot_compare_learning_curve(models, x, y, cv=cv, train_sizes=train_sizes, scoring=scoring,
                                title_prefix=title_prefix)


if __name__ == "__main__":
    x_cr, y_cr = load_credit_fraud_numpy(filepath="../data/creditcard.csv")
    x_sh, y_sh = load_shopper_intention_numpy(filepath='../data/online_shoppers_intention.csv')

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

    models_pruning_alpha = {
        'a.0'  : DecisionTreeClassifier(ccp_alpha=0.0, class_weight='balanced'),
        'a.001': DecisionTreeClassifier(ccp_alpha=0.001, class_weight='balanced'),
        'a.002': DecisionTreeClassifier(ccp_alpha=0.002, class_weight='balanced'),
        'a.003': DecisionTreeClassifier(ccp_alpha=0.003, class_weight='balanced')
    }

    c = 'entropy'
    models_balancing = {
        'balanced'  : DecisionTreeClassifier(criterion=c, class_weight='balanced'),
        'unbalanced': DecisionTreeClassifier(criterion=c)
    }

    train_sizes = np.linspace(0.2, 1, 7)

    # Show learning curve comparison of gini vs entropy
    # TODO: the P-R and ROC curves aren't too useful, why?
    compare_models_all_metrics(models_criterion, x_cr, y_cr, train_sizes=train_sizes, title_prefix="Credit Fraud")

    # Show learning curve comparison of balanced vs unbalanced
    compare_models_all_metrics(models_balancing, x_cr, y_cr, train_sizes=train_sizes, title_prefix="Credit Fraud")

    x_sub, x_val, y_sub, y_val = train_test_split(x_cr, y_cr, test_size=0.3, random_state=0)
    x_train, x_test, y_train, y_test = train_test_split(x_sub, y_sub, test_size=0.6, random_state=0)

    # Show how to determine an optimal ccpa value for post-pruning
    tree = DecisionTreeClassifier(criterion='entropy', class_weight='balanced')
    plot_nodes_vs_alpha(tree, x_train, y_train)
    plot_cost_complexity_pruning_path(tree, x_train, y_train)
    plot_acc_alpha_train_vs_test(tree, x_train, y_train, x_test, y_test)
