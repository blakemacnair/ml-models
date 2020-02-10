import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier

from credit_card_fraud.dataset import load_normalized_credit_fraud_numpy, load_credit_fraud_numpy
from online_shopper_intention.dataset import load_normalized_shopper_intention_numpy
from sklearn.model_selection import train_test_split

from metrics import compare_models_all_metrics

if __name__ == "__main__":
    # Load the data
    x_cr, y_cr = load_normalized_credit_fraud_numpy(filepath='../data/creditcard.csv')
    x_sh, y_sh = load_normalized_shopper_intention_numpy(filepath='../data/online_shoppers_intention.csv')
    train_sizes = np.linspace(0.3, 1, 8)

    # Generate some models to compare their learning efficacy
    models_kernels = dict(linear=SVC(kernel='linear'),
                          rbf=SVC(kernel='rbf'),
                          poly=SVC(kernel='poly'),
                          sigmoid=SVC(kernel='sigmoid'))

    # compare_models_all_metrics(models_kernels, x_sh, y_sh, train_sizes=train_sizes)

    models_classes = dict(rbf=SVC(kernel='rbf'),
                          linearSVC=LinearSVC(),
                          stochastic=SGDClassifier())

    # compare_models_all_metrics(models_classes, x_sh, y_sh, train_sizes=train_sizes)

    models_weights = dict(svc_1_5=SVC(kernel='linear', class_weight={0: 1, 1: 5}),
                          svc_1_100=SVC(kernel='linear', class_weight={0: 1, 1: 100}),
                          rbf_1_5=SVC(kernel='rbf', class_weight={0: 1, 1: 10}),
                          rbf_1_100=SVC(kernel='rbf', class_weight={0: 1, 1: 20}))

    # compare_models_all_metrics(models_weights, x_sh, y_sh, train_sizes=train_sizes)

    # Mess around with gamma in SVC with rbf kernel
    models_gamma = dict(gamma_scale=SVC(),
                        gamma_auto=SVC(gamma='auto'),
                        gamma5em6=SVC(gamma=5e-6),
                        gamma5em4=SVC(gamma=5e-4))

    # compare_models_all_metrics(models_gamma, x_sh, y_sh, train_sizes=train_sizes)

    # best model: {class='SVC', gamma='scale', kernel='rbf'} - all defaults ;)
    best_model = LinearSVC(class_weight='balanced')

    compare_models_all_metrics(models_classes, x_cr, y_cr, train_sizes, k_folds=5)
    print("Booty")
