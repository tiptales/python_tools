

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import numpy as np


#############################################################################
#                                                                           #
#                                 PROCESSING                                #
#                                                                           #
#############################################################################

def processit(tmp, y, scale=None, nan=None):

    """
    :param tmp: dataframe
    :param y: value of interest
    :param nan: if nan=mean, will handle missing values with mean and UNK
                if nan= null, nan=0 and UNK
                if nan= O2mean, tranforms 0 into mean of col
                :scale: type of scaler. "scaler" or "stdscaler"
    :return X_train, X_test, y_train, y_test
    """

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

    return X_train, X_test, y_train, y_test




#############################################################################
#                                                                           #
#                                 VIZ                                       #
#                                                                           #
#############################################################################


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
