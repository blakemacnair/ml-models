from time import time
import numpy as np
from sklearn.model_selection import RandomizedSearchCV


def random_search(clf, x, y, param_dist, scoring='f1', primary_scoring='f1', n_iter_search=10, n_top=3):
    # run randomized search
    rs = RandomizedSearchCV(clf, param_distributions=param_dist,
                            n_iter=n_iter_search,
                            scoring=scoring,
                            refit=primary_scoring,
                            n_jobs=-1)

    start = time()
    rs.fit(x, y)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    _report(rs.cv_results_, primary_scoring, n_top)


# Utility function to report best scores
def _report(results, primary_scoring='f1', n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_{}'.format(primary_scoring)] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation {0}: {1:.3f} (std: {2:.3f})"
                  .format(primary_scoring,
                          results['mean_test_{}'.format(primary_scoring)][candidate],
                          results['std_test_{}'.format(primary_scoring)][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
