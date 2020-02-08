import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import ShuffleSplit
from sklearn import preprocessing


from metrics import plot_compare_precision_recall_curve, plot_compare_learning_curve, plot_compare_roc_curve

from titanic.titanic_dataset import import_cleaned_titanic_data

if __name__ == "__main__":
    x_t, y_t, _, _ = import_cleaned_titanic_data(directorypath="titanic/")

    # https://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling
    x_t_scaled = preprocessing.scale(x_t)

    # models = {
    #     'LogReg': LogisticRegression(random_state=0),
    #     'LogRegCV': LogisticRegressionCV(random_state=0)  # Better performance marginally
    # }

    # models = {
    #     'liblinear': LogisticRegressionCV(random_state=0, solver='liblinear'),
    #     'newton-cg': LogisticRegressionCV(random_state=0, solver='newton-cg'),
    #     'lbfgs': LogisticRegressionCV(random_state=0, solver='lbfgs'),
    #     'sag': LogisticRegressionCV(random_state=0, solver='sag'),
    #     'saga': LogisticRegressionCV(random_state=0, solver='saga'),
    # }

    models = {
        'liblinear': LogisticRegressionCV(random_state=0, solver='liblinear'),
        'newton-cg': LogisticRegressionCV(random_state=0, solver='newton-cg'),
        'lbfgs'    : LogisticRegressionCV(random_state=0, solver='lbfgs')
    }

    # plot_compare_precision_recall_curve(models, x_m, y_m)
    # plot_compare_roc_curve(models, x_m, y_m)

    plot_compare_precision_recall_curve(models, x_t, y_t)
    plot_compare_roc_curve(models, x_t, y_t)
    plot_compare_learning_curve(models, x_t, y_t,
                                cv=ShuffleSplit(n_splits=5, test_size=0.4),
                                train_sizes=np.linspace(0.3, 1.0, 5))

    plot_compare_precision_recall_curve(models, x_t_scaled, y_t)
    plot_compare_roc_curve(models, x_t_scaled, y_t)
    plot_compare_learning_curve(models, x_t_scaled, y_t,
                                cv=ShuffleSplit(n_splits=5, test_size=0.4),
                                train_sizes=np.linspace(0.3, 1.0, 5))
