import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
import xgboost as xgb
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer, confusion_matrix, f1_score
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import average_precision_score


def plot_confusion_matrix(cm, classes, ax,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    print(cm)
    print('')

    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.sca(ax)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')


def trainit(tmp, y, idx, k, scale=None, nan=None):

    """

    :param tmp: dataframe
    :param y: value of interest
    :param idx: model index (for saving)
    :param k: nb of kfold validation. k should be >=0, <= n_features . If n_features < n_split, use k='all' to return all features.
                otherwise k.dtype = int
    :param nan: if nan=mean, will handle missing values with mean and UNK
                if nan= null, nan=0 and UNK
                if nan= O2mean, tranforms 0 into mean of col
                :scale: type of scaler. "scaler" or "stdscaler"
    :return nothing: write xgboost model and data to disk, create cm image
    """


    #############################################################################
    #                                                                           #
    #                                 PROCESSING                                #
    #                                                                           #
    #############################################################################

    tmp[str(y)] = tmp[str(y)].astype(object)
    objectlist = tmp.select_dtypes(include=['object']).columns.to_list()
    floatlist = tmp.select_dtypes(exclude=['object']).columns.to_list()

    if nan == 'mean':
        tmp[objectlist] = tmp[objectlist].fillna('UNK')
        tmp[floatlist] = tmp[floatlist].apply(lambda x: x.fillna(x.mean()), axis=0)

    if nan == 'null':
        tmp[objectlist] = tmp[objectlist].fillna('UNK')
        tmp[floatlist] = tmp[floatlist].fillna(0)

    if nan == 'O2mean':
        tmp[objectlist] = tmp[objectlist].fillna('UNK')
        tmp[floatlist] = tmp[floatlist].apply(lambda x: np.where(x == 0, x.mean(), x).tolist())

    na = tmp.isna().sum(axis=0)
    print('remaining nans: ', sum(na))


    for name in objectlist:
        col = pd.Categorical(tmp[name])
        tmp[name] = col.codes

    if scale == 'scaler':
        scaler = MinMaxScaler()
        tmp[floatlist] = scaler.fit_transform(tmp[floatlist])

    if scale == 'stdscaler':
        stdscaler = StandardScaler()
        tmp[floatlist] = stdscaler.fit_transform(tmp[floatlist])

    X_train, X_test, y_train, y_test = train_test_split(
        tmp.drop(str(y), 1),
        tmp[str(y)],
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=tmp[y]
    )


    #############################################################################
    #                                                                           #
    #                                 TRAINING                                  #
    #                                                                           #
    #############################################################################

    class_weights = np.sqrt(len(y_train[y_train == 0]) / len(y_train[y_train == 1]))

    # Pipeline
    pipe = Pipeline([
        ('fs', SelectKBest()),
        ('clf', xgb.XGBClassifier(objective='binary:logistic', scale_pos_weight= class_weights, seed=123))
    ])

    # Parameter Space
    search_space = [
    {
        'clf__n_estimators': [50, 100, 150, 200],
        'clf__learning_rate': [0.1, 0.2, 0.3],
        'clf__max_depth': range(3, 10),
        'clf__colsample_bytree': [i/10.0 for i in range(1, 3)],
        'clf__gamma': [i/10.0 for i in range(3)],
        'fs__score_func': [chi2],
        'fs__k': [k],
    }]

    # Cross validation
    kfold = KFold(n_splits=10, random_state=42, shuffle=True)

    # AUC and f1 as score
    scoring = {'AUC':'roc_auc', 'F1_aw': make_scorer(f1_score, average='weighted')}

    # Grid search
    grid = GridSearchCV(
        pipe,
        param_grid=search_space,
        cv=kfold,
        scoring=scoring,
        refit='F1_aw',
        verbose=1,
        n_jobs=-1
    )

    model = grid.fit(X_train, y_train)

    predict = model.predict(X_test)
    print('Best Score: {}'.format(model.best_score_))
    #print('AUC-mean: {}'.format(max(model['auc-mean'])))
    print('F1_aw: {}'.format(f1_score(y_test, predict)))
    cm = confusion_matrix(y_test,predict)

    # save model to file
    if not os.path.exists('models'):
        os.makedirs("models")
    writepath_ = os.path.join("models", str(idx))
    os.makedirs(writepath_)
    pickle.dump(model, open(os.path.join(writepath_, ("xgb_" + str(idx)) + ".dat"), "wb"))

    data_list = [X_train, X_test, y_train, y_test]
    pickle.dump(data_list, open(os.path.join(writepath_, ("data_" + str(idx)) + ".dat"), "wb"))

    fig, ax = plt.subplots()
    plot_confusion_matrix(cm, classes=np.unique(tmp[str(y)]),
                          ax=ax, title='xgboost.' + str(idx))
    plt.show()
    plt.savefig(os.path.join(writepath_, ("xgbplot_" + str(idx)) + ".png"))

    from sklearn.metrics import precision_recall_curve

    average_precision = average_precision_score(y_test, predict)
    print('Average precision-recall score: {0:0.2f}'.format(average_precision))
    disp = plot_precision_recall_curve(model, X_test, y_test)
    disp.ax_.set_title('2-class Precision-Recall curve: '
                       'AP={0:0.2f}'.format(average_precision))

# load model from file
# loaded_model = pickle.load(open("real_campain_2/models/xgboost1912_reduced_w_dummies.pickle.dat", "rb"))
# data_list1 = pickle.load(open("real_campain_2/models/data_xgboost1912_reduced_w_dummies.pickle.dat", "rb"))
# prd = loaded_model.predict(data_list1[1])
# print(confusion_matrix(data_list1[3],prd))

# Best score - max AUC
    # best_score = max(cv_results['auc-mean'])
    #
    # # Loss to be minimized
    # loss = 1 - best_score
    #
    # # Dictionary with information for evaluation
    # return {'loss': loss, 'params': params, 'status': STATUS_OK}