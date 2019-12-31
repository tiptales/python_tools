
"""
This script gives a quick overview of possible Classifiers results.
Only simple classifiers are tested with default or simple parameters, with no tuning.
i.e. CART is tested but not RF. If CART performed well, an ensemble model can be tested (Bagging, RF, etc.)
This test is intended to be fast, only the models performing well enough will be reviewed in detail.
"""


SEED = 124
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, f1_score, roc_auc_score, recall_score, confusion_matrix
import argparse
import os
import pickle
import utils as u

# A host of Scikit-learn models
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

#############################################################################
#                                                                           #
#                                 MODELING                                  #
#                                                                           #
#############################################################################

def get_models():
    """Generate a library of base learners."""
    svm_param = {'gamma': 'scale'}
    svm_ = svm.SVC(**svm_param)    ##no n_estimators
    logit_param = {'l1_ratio':0.5, 'penalty': 'elasticnet', 'solver': 'saga', 'random_state': SEED}
    logit = LogisticRegression(**logit_param)
    knn_param = {'n_neighbors': 3}
    knn = KNeighborsClassifier(**knn_param)  #no n-estimators
    nn_param = {'hidden_layer_sizes': (80, 10), 'early_stopping': True, 'random_state': SEED}  #no n_estimators
    nn = MLPClassifier(**nn_param)
    cart_param = {'criterion': 'gini'}
    cart = DecisionTreeClassifier(**cart_param)

    models = {'svm': svm_,
              'logit': logit,
              'knn': knn,
              'mlp-nn': nn,
              'cart': cart,
              }

    params = {'svm_param':svm_param, 'logit_param':logit_param, 'knn_param': knn_param, 'nn_param': nn_param, 'cart_param': cart_param}

    return models, params


def save(filename, model):
    pickle.dump(model, open(filename, 'wb'), protocol = 4)


def _predict(model_list, X_train, X_test, y_train, y_test):
    """Fit models in list on training set and return preds, save models to disk"""
    P = np.zeros((y_test.shape[0], len(model_list)))
    P = pd.DataFrame(P)
    suffix = 'sav'
    print("Fitting models.")
    cols = list()
    for i, (name, m) in enumerate(models.items()):
        print("%(n_)s , %(m_)s " % {'n_': name,'m_': m}, flush=False)
        m_ = m.fit(X_train, y_train)
        if not os.path.exists('models'):
            os.makedirs("models")
        save(os.path.join('models', name + "." + suffix), m_)
        P.iloc[:, i] = m_.predict(X_test)
        cols.append(name)
        print("done")

    P.columns = cols
    print("Done.\n")
    return P

def score_models(P, y):
    """Score model in prediction DF"""
    print("Scoring models.")

    prs = [precision_recall_fscore_support(y, P.loc[:, m], average='weighted') for m in P.columns]
    prsdict = defaultdict(list)
    for pr in prs:
        prsdict['precision'].append(pr[0])
        prsdict['recall'].append(pr[1])
        prsdict['fbeta_score'].append(pr[2])
        prsdict['support'].append(pr[3])
    rc = [recall_score(y, P.loc[:, m]) for m in P.columns]
    f1 = [f1_score(y, P.loc[:, m], average='weighted') for m in P.columns]
    roc = [roc_auc_score(y, P.loc[:, m], average='weighted') for m in P.columns]
    cm = [confusion_matrix(y, P.loc[:, m]) for m in P.columns]
    cmdict= defaultdict(list)
    for c in cm:
        cmdict['TP'].append(c[1, 1])
        cmdict['TN'].append(c[0, 0])
        cmdict['FP'].append(c[0, 1])
        cmdict['FN'].append(c[1, 0])
    names = pd.Series(['precision', 'recall', 'fbeta-score', 'support', 'recall-score', 'f1-weighted', 'roc', 'TP', 'TN', 'FP', 'FN'])
    t = pd.DataFrame(data=[prsdict['precision'], prsdict['recall'],prsdict['fbeta_score'], prsdict['support'], rc, f1, roc, cmdict['TP'], cmdict['TN'], cmdict['FP'], cmdict['FN']])
    t.columns = P.columns
    t['measures'] = names.values

    return t



if __name__ == '__main__':


    parser = argparse.ArgumentParser()

    parser.add_argument('--infile', type=str, help='input file path')
    parser.add_argument('--Y', type=str, help='value of interest')
    parser.add_argument('--sep', type=str, help='pandas sep if needed')

    args = parser.parse_args()
    dataf = pd.read_csv(args.infile, args.sep)
    y = args.Y
    print(dataf.head())
    #dataf[y] = dataf[y].astype(object)
    print(dataf.dtypes)

    X_train, X_test, y_train, y_test = u.processit(dataf, y = args.Y)
    models, params = get_models()
    P = _predict(models, X_train, X_test, y_train, y_test)
    sc = score_models(P, y_test)

    sc.to_csv('models/measures.csv')


    def print_full(x):
        pd.set_option('display.max_columns', x.shape[1])
        print(x)
        pd.reset_option('display.max_columns')
    print_full(sc)

