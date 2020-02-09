from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.model_selection import ShuffleSplit
import numpy as np

from credit_card_fraud.dataset import load_credit_fraud_numpy, load_normalized_credit_fraud_numpy
from online_shopper_intention.dataset import load_shopper_intention_numpy, load_normalized_shopper_intention_numpy

from metrics import compare_models_all_metrics
from sklearn.model_selection import cross_val_predict, cross_validate, train_test_split

from time import time
import scipy.stats as stats
from sklearn.utils.fixes import loguniform

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_f1'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation f1: {0:.3f} (std: {1:.3f})"
                  .format(results['mean_test_f1'][candidate],
                          results['std_test_f1'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


if __name__ == "__main__":
    x_cr, y_cr = load_credit_fraud_numpy(filepath='./data/creditcard.csv')
    x_cr_norm, y_cr_norm = load_normalized_credit_fraud_numpy(filepath='./data/creditcard.csv')
    x_sh, y_sh = load_normalized_shopper_intention_numpy(filepath='./data/online_shoppers_intention.csv')

    # Distance comparison
    clfs_distance = dict(uniformWeight=KNeighborsClassifier(),
                         distanceWeight=KNeighborsClassifier(weights='distance'))

    # Radius based vs vanilla
    clfs_class = dict(vanilla=KNeighborsClassifier(),
                      radius100=RadiusNeighborsClassifier(outlier_label='most_frequent',
                                                          weights='distance'))

    # N neighbors comparison
    clfs_neighbors = dict()
    for i in range(5, 51, 5):
        clfs_neighbors['{}-neighbor'.format(i)] = KNeighborsClassifier(n_neighbors=i, weights='distance')

    # Algorithm comparison
    clfs_algorithm = dict(auto=KNeighborsClassifier(algorithm='auto', weights='distance'),
                          ball_tree=KNeighborsClassifier(algorithm='ball_tree', weights='distance'),
                          kd_tree=KNeighborsClassifier(algorithm='kd_tree', weights='distance'),
                          brute=KNeighborsClassifier(algorithm='brute', weights='distance'))

    train_sizes = np.linspace(0.1, 1, 10)
    # compare_models_all_metrics(clfs_distance, x_sh, y_sh, train_sizes=train_sizes, title_prefix="Shopper Intention")
    # compare_models_all_metrics(clfs_neighbors, x_sh, y_sh, train_sizes=train_sizes, title_prefix="Shopper Intention")

    # specify parameters and distributions to sample from
    param_dist = {
        'n_neighbors': range(30, 56, 5),
        'leaf_size'  : range(30, 101, 20),
        'p'          : range(3, 9, 1),
        'algorithm'  : ['ball_tree', 'kd_tree', 'brute']
    }

    scoring = ['roc_auc', 'balanced_accuracy', 'f1']

    # run randomized search
    clf = KNeighborsClassifier(weights='distance')
    n_iter_search = 10
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       n_iter=n_iter_search,
                                       scoring=scoring,
                                       refit='f1',
                                       n_jobs=-1)

    start = time()
    random_search.fit(x_sh, y_sh)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.cv_results_)

    """
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

    print("Booty - Done")
