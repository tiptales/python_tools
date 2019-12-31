import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import utils as u
import argparse

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, confusion_matrix, f1_score, recall_score, precision_recall_curve, auc,roc_curve, roc_auc_score,classification_report
import warnings
warnings.filterwarnings('ignore')


def under(df, sampling, y, maj_, min_):
    """
    :param tmp: dataframe
    :param maj: group for majority class. Is turned to 0
    :param min_: groump for minority class. Is turned to 1
    :computed norm_idx: index of exceptional events in tmp
    :computed except_idx: index of normal events in tmps
    :param sampling : proportion of normal event to keep: will be resumed to normal = sampling*exception
    :return a dataset taking only a fraction of normal events and the whole exceptional events.
    """
    tmp = deepcopy(df)
    tmp[y] = tmp[y].replace([maj_], 0)
    tmp[y] = tmp[y].replace([min_], 1)
    print('df.head()', tmp[y].head(), 'type', tmp[y].dtype)

    except_idx = np.array(tmp[tmp[str(y)] == 1].index)
    norm_idx = np.array(tmp[tmp[str(y)]== 0].index)
    norm = len(tmp[tmp[str(y)] == 0])
    print('length of norm: ', norm)
    exception = len(tmp[tmp[(str(y))]== 1])
    print('length of except: ', exception)
    print("percentage of normal events is", (norm / (norm + exception))*100)
    print("percentage of exceptional events", (exception/(norm+exception))*100)

    norm_idx_under = np.array(np.random.choice(norm_idx, (sampling*exception), replace=False))
    under_data = np.concatenate([except_idx, norm_idx_under])
    under_data = tmp.iloc[under_data, :]
    under_data[str(y)] = under_data[str(y)].astype(object)
    print("normal events proportion is now:", len(under_data[under_data[str(y)] == 0]) / len(under_data[str(y)]))
    print("exceptional events proportion is now :",
          len(under_data[under_data[str(y)] == 1]) / len(under_data[str(y)]))
    print("total number of record in resampled data is:", len(under_data[str(y)]))

    return (under_data)


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
    u.plot_confusion_matrix(cm, classes=np.unique(ytrain), ax=ax, title='resampled logit.')
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

    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type =str, help = 'input file path')
    parser.add_argument('--sep', type =str, help = 'pandas sep if needed')

    parser.add_argument('--Y', type = str, help='value of interest')
    parser.add_argument('--majority',help='majority class label in the dataset, will be turned to 0')
    parser.add_argument('--minority', help='minority class label in the dataset, will be turned to 1')

    args = parser.parse_args()
    dataf = pd.read_csv(args.infile, args.sep)
    Y, maj_, min_ = args.Y, args.majority, args.minority


    for i in range(1,3):
        print('____________________________________results on train data________________________________________________')
        print("the under data for {} proportion".format(i))
        print()
        under_data = under(dataf, i, Y, maj_=maj_, min_=min_)
        print("------------------------------------------------------------")
        print()
        print("the model classification for {} proportion".format(i))
        print()
        under_Xtrain, under_Xtest, under_ytrain, under_ytest = u.processit(under_data, Y, scale= 'stdscaler', nan=None)

        print()
        clf = LogisticRegression()
        #model(clf, under_Xtrain, under_Xtest, under_ytrain, under_ytest)
        print("________________________________________________________________________________________________________")
        # print('____________________________________results on test data________________________________________________')
        data_Xtrain, data_Xtest, data_ytrain, data_ytest = u.processit(dataf, Y)
        model(clf, under_Xtrain, data_Xtest, under_ytrain, data_ytest)


