from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.base import copy

from credit_card_fraud.dataset import load_credit_fraud_numpy, load_normalized_credit_fraud_numpy
from online_shopper_intention.dataset import load_shopper_intention_numpy, load_normalized_shopper_intention_numpy

from metrics import compare_models_all_metrics

if __name__ == "__main__":
    # Load the data
    x_cr, y_cr = load_credit_fraud_numpy(filepath='./data/creditcard.csv')
    x_cr_norm, y_cr_norm = load_normalized_credit_fraud_numpy(filepath='./data/creditcard.csv')
    x_sh, y_sh = load_normalized_shopper_intention_numpy(filepath='./data/online_shoppers_intention.csv')

    # Start by running randomized search to give us a good ballpark for an optimal configuration for the classifier
    # specify parameters and distributions to sample from
    param_dist = dict(n_neighbors=range(30, 56, 5),
                      leaf_size=range(30, 101, 20),
                      p=range(3, 9, 1),
                      algorithm=['ball_tree', 'kd_tree', 'brute'])

    scoring = ['roc_auc', 'balanced_accuracy', 'f1']

    clf = KNeighborsClassifier(weights='distance')
    n_iter_search = 10
    # random_search(clf, x_sh, y_sh, param_dist=param_dist, scoring=scoring, primary_scoring='f1')
    """ random_search output:
RandomizedSearchCV took 233.13 seconds for 10 candidates parameter settings.

Model with rank: 1
Mean validation f1: 0.501 (std: 0.070)
Parameters: {'p': 7, 'n_neighbors': 30, 'leaf_size': 50, 'algorithm': 'brute'}

Model with rank: 2
Mean validation f1: 0.500 (std: 0.068)
Parameters: {'p': 8, 'n_neighbors': 35, 'leaf_size': 70, 'algorithm': 'kd_tree'}

Model with rank: 3
Mean validation f1: 0.498 (std: 0.072)
Parameters: {'p': 7, 'n_neighbors': 35, 'leaf_size': 70, 'algorithm': 'kd_tree'}
    """

    # Init the best model to base rest of tests on
    base = KNeighborsClassifier(p=8, n_neighbors=35, leaf_size=70, algorithm='kd_tree', weights='distance', n_jobs=-1)

    train_sizes = np.linspace(0.1, 1, 10)

    # Compare different n_neighbors values
    clfs_neighbors = dict()
    for n_neighbors in range(15, 56, 5):
        clf = copy.deepcopy(base)
        clf.n_neighbors = n_neighbors
        clfs_neighbors['{}-nn'.format(n_neighbors)] = clf

    compare_models_all_metrics(clfs_neighbors, x_sh, y_sh, train_sizes=train_sizes, title_prefix="Shopper Intention")

    # Compare different leaf sizes
    clfs_leaf_size = dict()
    for leaf_size in range(30, 81, 5):
        clf = copy.deepcopy(base)
        clf.leaf_size = leaf_size
        clfs_neighbors['{}-neighbor'.format(leaf_size)] = clf

    compare_models_all_metrics(clfs_leaf_size, x_sh, y_sh, train_sizes=train_sizes, title_prefix="Shopper Intention")

    # Compare different p values
    clfs_p = dict()
    for p in range(30, 81, 5):
        clf = copy.deepcopy(base)
        clf.p = p
        clfs_neighbors['{}-neighbor'.format(p)] = clf

    compare_models_all_metrics(clfs_p, x_sh, y_sh, train_sizes=train_sizes, title_prefix="Shopper Intention")
    print("Booty - Done")
