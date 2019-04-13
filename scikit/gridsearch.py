

from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


import time
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

import logging
import os



#comment added
#create the top-level parser
#
# parser = argparse.ArgumentParser()
# subparsers = parser.add_subparsers()
# FUNCTION_MAP = {'search_mlp': mlp, 'search_xgb': xgb, 'search_rf': rf}
# parser.add_argument('command', choices=FUNCTION_MAP.keys())
# args = parser.parse_args()

# #############################################################################
#                         DATA   #load   OR  #simulate
# #############################################################################
#load
X_train=np.load('data/out/training/X_train.dat')
Y_train=np.load('data/out/training/Y_train.dat')
X_test=np.load('data/out/training/X_test.dat')
Y_test=np.load('data/out/training/Y_test.dat')

# # #simulate
# def MackayGlass(y, N):
#     b = 0.1
#     c = 0.2
#     tau = 17
#     for n in range(17, N):
#         y = np.append(y, (y[n] - b * y[n] + (c * y[n - tau] / (1 + ((y[n - tau]) ** 10)))))
#
#     return (y)
#
# y_list = [0.9697, 0.9699, 0.9794, 1.0003, 1.0319, 1.0703, 1.1076, 1.1352, 1.1485, 1.1482, 1.1383, 1.1234, 1.1072,
#           1.0928, 1.0820, 1.0756, 1.0739, 1.0759]
# Y = MackayGlass(y_list, 699)
# Y_train = Y[0:300].reshape(-1, 1)
# Y_test = Y[301:700].reshape(-1, 1)
# X_train = np.arange(0, 300, 1).reshape(-1, 1)
# X_test = Y_test[0].reshape(1, -1)

# #############################################################################
#                                  FUNCTIONS
# #############################################################################
#TODO : make utils.py

# set logger
def set_logger(log_path):
    """ he is saved to `data/out/models/GS.log`.
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


# save to disk
def save(filename, model):
    pickle.dump(model, open(filename, 'wb'))

# Grid Options ! MLP ! RF ! GBR !
def search(scikitObj, X_train, Y_train, X_test, Y_test):
    t1 = time.time()
    if scikitObj == 'MLP':
        set_logger(os.path.join('data/out/models', 'mlp_gs.log'))
        mlp = MLPRegressor()
        param_grid = {'hidden_layer_sizes': [i for i in range(1,50)],
              'activation': ['logistic', 'tanh', 'relu'],
              'solver': ['lbfgs', 'adam', 'sgd'],
              'alpha': [0.01, 0.001],
              'max_iter': [1000],
              'random_state': [1],
              'early_stopping': [True],
              'warm_start': [True],
              'n-jobs': [-1]}
        MLP_GS = GridSearchCV(mlp, param_grid=param_grid,
                   verbose=True, pre_dispatch='2*n_jobs')

        MLP_GS.fit(X_train, Y_train)
        t2 = time.time()
        print(' ')
        print((t2-t1)*1000)
        print(' ')
        print(MLP_GS.best_params_)
        print(' ')
        print(MLP_GS.cv_results_.keys())
        print(' ')
        optimised_MLP = MLP_GS.best_estimator_
        optimised_MLP_instance = optimised_MLP.fit(X_train, Y_train)
        Y_pred = optimised_MLP_instance.predict(X_test)
        save('data/out/models/mlp.model.sav', MLP_GS)
        save('data/out/models/mlp.fit.sav', optimised_MLP_instance)

    if scikitObj =='RF':
        set_logger(os.path.join('data/out/models', 'rf_gs.log'))
        rf = RandomForestRegressor()
        param_grid = {'bootstrap': [True, False],
                    'criterion': ['mse', 'mae'],
                    'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                    'max_features': ['auto', 'sqrt', 'log2'],
                    'min_samples_leaf': [i for i in range(1, 10)],
                    'min_samples_split': [i for i in range(2, 10)],
                    'n_estimators': [i for i in range(1, 500)],
                    'oob_score': [True],
                    'warm_start': [True]}
        RF_GS = GridSearchCV(rf, param_grid=param_grid,
                              verbose=True, cv = 5, pre_dispatch='2*n_jobs')

        RF_GS.fit(X_train, Y_train)
        t2 = time.time()
        print(' ')
        print((t2 - t1) * 1000)
        print(' ')
        print(RF_GS.best_params_)
        print(' ')
        print(RF_GS.cv_results_.keys())
        print(' ')
        optimised_RF = RF_GS.best_estimator_
        optimised_RF_instance = optimised_RF.fit(X_train, Y_train)
        Y_pred = optimised_RF_instance.predict(X_test)
        save('data/out/models/rf.model.sav', RF_GS)
        save('data/out/models/rf.fit.sav', optimised_RF_instance)

    t1 = time.time()
    if scikitObj == 'GBR':
        set_logger(os.path.join('data/out/models', 'gbr_gs.log'))
        gbr = GradientBoostingRegressor()
        param_grid = {'loss': ['ls', 'lad', 'huber', 'quantile'],
                      'learning_rate': [0.1, 0.01, 0.001],
                      'n_estimators': [100, 200, 300, 400, 500],
                      'criterion': ['friedman_mse', 'mse', 'mae'],
                      'max_depth': [10, 50, 100]}
        GBR_GS = GridSearchCV(gbr, param_grid=param_grid,
                              verbose=10, n_jobs = 6, pre_dispatch='2*n_jobs')

        GBR_GS.fit(X_train, Y_train)
        t2 = time.time()
        print(' ')
        print((t2 - t1) * 1000)
        print(' ')
        print(GBR_GS.best_params_)
        print(' ')
        print(GBR_GS.cv_results_.keys())
        print(' ')
        optimised_GBR = GBR_GS.best_estimator_
        optimised_GBR_instance = optimised_GBR.fit(X_train, Y_train)
        Y_pred = optimised_GBR_instance.predict(X_test)
        save('data/out/models/gbr.model.sav', GBR_GS)
        save('data/out/models/gbr.fit.sav', optimised_GBR_instance)





# test it
if __name__ == '__main__':

    #search('MLP', X_train, Y_train, X_test, Y_test)
    #search('RF', X_train, Y_train, X_test, Y_test)
    search('GBR', X_train, Y_train, X_test, Y_test)

    # parse the args and call whatever function was selected
    #func = FUNCTION_MAP[args.command]
    #func()
