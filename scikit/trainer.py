SEED = 124

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from math import sqrt
import os
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import roc_auc_score

# A host of Scikit-learn models
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor


####  LOAD DATA   ###
df = pd.read_csv('training.csv', header=0, index_col=False)

#define training and test sets
msk = np.random.rand(len(df)) < 0.8
train = df[msk].drop('value_of_interest', axis=1)
test = df[~msk].drop('value_of_interest', axis=1)
train.head()

# TODO argparse input/outputs
# TODO makedirs
# TODO pickle instead of dat
# TODO dask grid search : https://dask-searchcv.readthedocs.io/en/latest/
# TODO flask
# TODO write tests
# TODO from FutureWarning: Method as.matrix will be removed in a future version. Use .values instead.
X_train = train.as_matrix()
X_train.dump('training/X_train.dat')
Y_train = (df[msk].as_matrix())
Y_train.dump('training/Y_train.dat')
X_test = test.as_matrix()
X_test.dump('training/X_test.dat')
Y_test = (df[~msk]).as_matrix()
Y_test.dump('training/Y_test.dat')
#
#
# X_train=np.load('data/out/training/X_train.dat')
# Y_train=np.load('data/out/training/Y_train.dat')
# X_test=np.load('data/out/training/X_test.dat')
# Y_test=np.load('data/out/training/Y_test.dat')

#np.isnan(np.sum(Y_train))


def save(filename, model):
    pickle.dump(model, open(filename, 'wb'), protocol = 4)
#
# save('data/out/models/xgb.tmp.model.sav', xgb)

def get_models():
    """Generate a library of base learners."""
    svm_param = {'gamma': 'scale'}
    svm_ = svm.SVR(**svm_param)    ##no n_estimators
    knn_param = {'n_neighbors': 3}
    knn = KNeighborsRegressor(**knn_param)  #no n-estimators
    nn_param = {'hidden_layer_sizes': (80, 10), 'early_stopping': True, 'random_state': SEED}  #no n_estimators
    nn = MLPRegressor(**nn_param)
    gbx_param = {'n_estimators': 100, 'random_state':SEED}
    gbx = GradientBoostingRegressor(**gbx_param)
    cart_param = {'criterion': 'mse'}
    cart = DecisionTreeRegressor(**cart_param)

    models = {'svm': svm_,
              'knn': knn,
              'mlp-nn': nn,
              'gbm': gbx,
              'cart': cart,
              }

    params = {'svm_param':svm_param, 'knn_param': knn_param, 'nn_param': nn_param, 'gbx_param': gbx_param, 'cart_param': cart_param}

    return models, params


def train_predict(model_list):
    """Fit models in list on training set and return preds"""
    P = np.zeros((Y_test.shape[0], len(model_list)))
    P = pd.DataFrame(P)
    suffix = 'sav'
    print("Fitting models.")
    cols = list()
    for i, (name, m) in enumerate(models.items()):
        print("%(n_)s , %(m_)s " % {'n_': name,'m_': m}, flush=False)
        m_ = m.fit(X_train, Y_train)
        save(os.path.join('data/out/models', name + "." + suffix), m_)
        P.iloc[:, i] = m_.predict(X_test)
        cols.append(name)
        print("done")

    P.columns = cols
    print("Done.\n")
    return P

def score_models(P, y):
    """Score model in prediction DF"""
    print("Scoring models.")

    ve = [explained_variance_score(y, P.loc[:, m]) for m in P.columns]
    mse = [mean_squared_error(y, P.loc[:, m]) for m in P.columns]
    r2 = [r2_score(y, P.loc[:, m]) for m in P.columns]
    names = pd.Series(['ve', 'mse', 'r2'])
    t = pd.DataFrame(data=[ve, mse, r2])
    t.columns = P.columns
    t['measures'] = names.values
    print(t)
    return t

models , params = get_models()
#params['svm_param']['gamma']
P = train_predict(models)
sc = score_models(P, Y_test)

sc.to_csv('models/measures.csv')






