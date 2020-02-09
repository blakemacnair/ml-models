from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ShuffleSplit
import numpy as np

from titanic.titanic_dataset import import_cleaned_titanic_data
from online_shopper_intention.dataset import load_shopper_intention_numpy

from metrics import plot_compare_precision_recall_curve, plot_compare_roc_curve, plot_compare_learning_curve


if __name__ == "__main__":
    cv = ShuffleSplit(n_splits=7, test_size=0.3, random_state=0)

    # Algorithm comparison
    clf_auto = KNeighborsClassifier(algorithm='auto', weights='distance')
    clf_ball_tree = KNeighborsClassifier(algorithm='ball_tree', weights='distance')
    clf_kd_tree = KNeighborsClassifier(algorithm='kd_tree', weights='distance')
    clf_brute = KNeighborsClassifier(algorithm='brute', weights='distance')

    classifiers = {
        'auto': clf_auto,
        'ball_tree': clf_ball_tree,
        'kd_tree': clf_kd_tree,
        'brute': clf_brute
    }

    x, y, x_test, test_ids = import_cleaned_titanic_data(directorypath="titanic/")

    plot_compare_precision_recall_curve(classifiers=classifiers, x=x, y=y)
    plot_compare_roc_curve(classifiers=classifiers, x=x, y=y)
    plot_compare_learning_curve(classifiers=classifiers, x=x, y=y, cv=cv,
                                train_sizes=np.linspace(0.1, 1.0, 15))

    x, y = load_shopper_intention_numpy(filepath='../data/online_shoppers_intention.csv')

    plot_compare_precision_recall_curve(classifiers=classifiers, x=x, y=y)
    plot_compare_roc_curve(classifiers=classifiers, x=x, y=y)
    plot_compare_learning_curve(classifiers=classifiers, x=x, y=y, cv=cv,
                                train_sizes=np.linspace(0.1, 1.0, 15))

    print("Booty - Done")
