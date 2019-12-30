import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
from copy import deepcopy
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, confusion_matrix, f1_score, recall_score, precision_recall_curve, auc,roc_curve, roc_auc_score,classification_report
import warnings
warnings.filterwarnings('ignore')

#TODO quote source
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


def under(df, sampling, y):
    """
    :param tmp: dataframe
    :computed norm_idx: index of exceptional events in tmp
    :computed except_idx: index of normal events in tmps
    :param sampling : proportion of normal event to keep: will be resumed to normal = sampling*exception
    :return a dataset taking only a fraction of normal events and the whole exceptional events.
    """
    tmp = deepcopy(df)
    except_idx = np.array(tmp[tmp[str(y)] == 1].index)
    norm_idx = np.array(tmp[tmp[str(y)]== 0].index)
    norm = len(tmp[tmp[str(y)] == 0])
    exception = len(tmp[tmp[(str(y))]==1])
    print("percentage of normal eventsn is", (norm / (norm + exception))*100)
    print("percentage of exceptional events", (exception/(norm+exception))*100)

    norm_idx_under = np.array(np.random.choice(norm_idx, (sampling*exception), replace=False))
    under_data = np.concatenate([except_idx, norm_idx_under])
    under_data = tmp.iloc[under_data, :]
    under_data[str(y)] = under_data[str(y)].astype(object)
    print("the normal events proportion is :", len(under_data[under_data[str(y)] == 0]) / len(under_data[str(y)]))
    print("the exceptional event proportion is :",
          len(under_data[under_data[str(y)] == 1]) / len(under_data[str(y)]))
    print("total number of record in resampled data is:", len(under_data[str(y)]))

    return (under_data)


def stdandscale(tmp, y):

    floatlist = tmp.select_dtypes(exclude=['object']).columns.to_list()
    stdscaler = StandardScaler()
    tmp[floatlist] = stdscaler.fit_transform(tmp[floatlist])

    objectlist = tmp.select_dtypes(include=['object']).columns.to_list()
    if not str(y) in objectlist:
        objectlist = objectlist + [str(y)]
    for name in objectlist:
        col = pd.Categorical(tmp[name])
        tmp[name] = col.codes

    return tmp


def processing(df, y):
    """
    :param x: dataframe with predictors and value of interest
    """
    tmp_ = deepcopy(df)
    tmp = stdandscale(tmp_, y)
    X_train, X_test, y_train, y_test = train_test_split(
        tmp.drop(str(y), 1),
        tmp[str(y)],
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=tmp[y]
    )
    print("length of training data is : ", len(X_train))
    print("length of test data is : ", len(X_test))
    print('y_train dtype is ', y_train.dtype, 'y_test dtype is : ', y_test.dtype)
    return(X_train, X_test, y_train, y_test)


def model(model, Xtrain, Xtest, ytrain, ytest):
    """
    :param model: estimator
    """
    clf = model
    clf.fit(Xtrain, ytrain)
    pred = clf.predict(Xtest)
    cm = confusion_matrix(ytest, pred)
    print('############################')
    print("the recall for this model is :", cm[1, 1]/(cm[1, 1]+cm[1, 0]))
    print('############################')
    fig, ax = plt.subplots()
    plot_confusion_matrix(cm, classes=np.unique(ytrain), ax=ax, title='resampled logit.')
    plt.show()
    #fig = plt.figure(figsize=(6,3))
    print("TP: ", cm[1, 1,], "exceptional events transaction predicted exception") #
    print("TN: ", cm[0, 0], "normal events predicted normal")
    print("FP: ", cm[0, 1], "normal events predicted exception")
    print("FN: ", cm[1, 0], "exceptional events predicted normal")
    # sns.heatmap(cm, cmap="coolwarm_r", annot=True, linewidths=0.5)
    # plt.title("Confusion_matrix")
    # plt.xlabel("Predicted_class")
    # plt.ylabel("Real class")
    # plt.show()
    print("\n----------Classification Report------------------------------------")
    print(classification_report(ytest, pred))



if __name__ == '__main__':

    #TODO dataf = (makedefault)
    from random import choices
    dataf_ = pd.DataFrame({'Q1': choices([1, 2, 3, 4, 5], k=30), 'Q2': choices([6, 7, 8, 9, 10], k=30), 'Q3': choices(['A', 'B', 'C', 'D', 'E'], k=30), 'Y': choices([0, 0,0,0,0, 1], k= 30)})
    for i in range(1,3):
        print('____________________________________results on train data________________________________________________')
        print("the under data for {} proportion".format(i))
        print()
        under_data = under(dataf_, i, 'Y')
        print("------------------------------------------------------------")
        print()
        print("the model classification for {} proportion".format(i))
        print()
        under_Xtrain, under_Xtest, under_ytrain, under_ytest = processing(under_data, 'Y')
        print()
        clf = LogisticRegression()
        #model(clf, under_Xtrain, under_Xtest, under_ytrain, under_ytest)
        print("________________________________________________________________________________________________________")
        # print('____________________________________results on test data________________________________________________')
        data_Xtrain, data_Xtest, data_ytrain, data_ytest = processing(dataf_, 'Y')
        model(clf, under_Xtrain, data_Xtest, under_ytrain, data_ytest)


