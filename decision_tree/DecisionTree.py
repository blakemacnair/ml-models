import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

from online_shopper_intention.dataset import load_shopper_intention_numpy
from credit_card_fraud.dataset import load_credit_fraud_numpy

from metrics import plot_nodes_vs_alpha, plot_cost_complexity_pruning_path, plot_acc_alpha_train_vs_test
from metrics import compare_models_all_metrics
from sklearn.model_selection import cross_val_predict, cross_validate, train_test_split

if __name__ == "__main__":
    x_cr, y_cr = load_credit_fraud_numpy(filepath="../data/creditcard.csv")
    x_sh, y_sh = load_shopper_intention_numpy(filepath='../data/online_shoppers_intention.csv')
    c = 'entropy'
    a = 1e-2

    models_criterion = {
        'gini'   : DecisionTreeClassifier(criterion='gini', class_weight='balanced', ccp_alpha=a),
        'entropy': DecisionTreeClassifier(criterion='entropy', class_weight='balanced', ccp_alpha=a)
    }

    models_balancing = dict(balanced=DecisionTreeClassifier(criterion=c, class_weight='balanced'),
                            unbalanced=DecisionTreeClassifier(criterion=c))

    ensembles = dict(tree=DecisionTreeClassifier(criterion='entropy', class_weight='balanced', ccp_alpha=a),
                     randomForest=RandomForestClassifier(),
                     bagging=BaggingClassifier(DecisionTreeClassifier(criterion='entropy',
                                                                      class_weight='balanced',
                                                                      ccp_alpha=a)))

    train_sizes = np.linspace(0.2, 1, 7)

    # Show learning curve comparison of gini vs entropy
    # TODO: the P-R and ROC curves aren't too useful, why?
    # compare_models_all_metrics(models_criterion, x_cr, y_cr, train_sizes=train_sizes, title_prefix="Credit Fraud")

    # Show learning curve comparison of balanced vs unbalanced
    # compare_models_all_metrics(models_balancing, x_cr, y_cr, train_sizes=train_sizes, title_prefix="Credit Fraud")

    compare_models_all_metrics(ensembles, x_cr, y_cr, train_sizes=train_sizes, title_prefix="Credit Fraud")

    x_sub, x_val, y_sub, y_val = train_test_split(x_cr, y_cr, test_size=0.3, random_state=0)
    x_train, x_test, y_train, y_test = train_test_split(x_sub, y_sub, test_size=0.6, random_state=0)

    # Show how to determine an optimal ccpa value for post-pruning
    # tree = DecisionTreeClassifier(criterion='entropy', class_weight='balanced')
    # plot_nodes_vs_alpha(tree, x_train, y_train)
    # plot_cost_complexity_pruning_path(tree, x_train, y_train)
    # plot_acc_alpha_train_vs_test(tree, x_train, y_train, x_test, y_test)
