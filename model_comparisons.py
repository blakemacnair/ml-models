from credit_card_fraud.dataset import load_credit_fraud_numpy
from online_shopper_intention.dataset import load_shopper_intention_numpy

import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from neural_network.NeuralNet import skorch_murder_bot
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

from metrics import compare_models_all_metrics

if __name__ == "__main__":
    x_cr, y_cr = load_credit_fraud_numpy(filepath='./data/creditcard.csv')
    x_sh, y_sh = load_shopper_intention_numpy(filepath='./data/online_shoppers_intention.csv')
    train_sizes = np.linspace(0.3, 0.8, 9)

    x_cr = x_cr.astype(np.float32)
    y_cr = y_cr.astype(np.long)
    cr_input_size = x_cr.shape[1]

    x_sh = x_sh.astype(np.float32)
    y_sh = y_sh.astype(np.long)
    sh_input_size = x_sh.shape[1]

    models_cr = dict(
        knn=KNeighborsClassifier(p=8, n_neighbors=35, leaf_size=70, algorithm='kd_tree'),
        mlp=skorch_murder_bot(input_size=cr_input_size,
                              hidden_layers=np.array([cr_input_size * 2, cr_input_size // 2])),
        svm=LinearSVC(class_weight='balanced'),
        baggingTrees=BaggingClassifier(DecisionTreeClassifier(criterion='entropy', ccp_alpha=0.005)),
        boosting=GradientBoostingClassifier(ccp_alpha=0.005)
    )

    compare_models_all_metrics(models_cr, x_cr, y_cr, train_sizes)

    models_sh = dict(
        knn=KNeighborsClassifier(p=8, n_neighbors=35, leaf_size=70, algorithm='kd_tree'),
        mlp=skorch_murder_bot(input_size=sh_input_size,
                              hidden_layers=np.array([sh_input_size * 2, sh_input_size // 2])),
        svm=LinearSVC(class_weight='balanced'),
        baggingTrees=BaggingClassifier(DecisionTreeClassifier(criterion='entropy', ccp_alpha=0.005)),
        boosting=GradientBoostingClassifier(ccp_alpha=0.005)
    )

    compare_models_all_metrics(models_sh, x_sh, y_sh, train_sizes)

    print("Dope")
