import numpy as np

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report

from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier

from online_shopper_intention.dataset import load_shopper_intention_numpy
from credit_card_fraud.dataset import load_credit_fraud_numpy

from metrics import compare_models_all_metrics

if __name__ == "__main__":
    models = {
        'Adaboost': AdaBoostClassifier(),
        'Gradient': GradientBoostingClassifier(),
        'HistGrad': HistGradientBoostingClassifier()
    }

    x_cr, y_cr = load_credit_fraud_numpy(filepath="./data/creditcard.csv")
    x_sh, y_sh = load_shopper_intention_numpy(filepath='./data/online_shoppers_intention.csv')

    train_sizes = np.linspace(0.3, 1, 5)
    compare_models_all_metrics(models, x_cr, y_cr, train_sizes)

    print("Booty")
