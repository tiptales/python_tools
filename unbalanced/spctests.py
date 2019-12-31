
#############################################################################
#                                                                           #
#                       STATISTICAL PROCESS CONTROL                         #
#                                                                           #
#############################################################################


import pandas as pd
import resampling.sample as sp
import features_engineering.covariates as ft
import utils as u



dataf = pd.read_csv('data/bank/bank.csv', sep=';')
y = 'y'
y2 = 'job'

print(dataf.dtypes)
print('__________________________________________________________________________')
print(ft.pairwisetest(dataf, y))
print('__________________________________________________________________________')
print(ft.chi2(dataf))
print('__________________________________________________________________________')
print(ft.corrcoef(dataf, method='spearman'))
print('__________________________________________________________________________')
[print(i, '\n') for i in ft.aovtables(dataf, y2, y)]
print('__________________________________________________________________________')
print(ft.XDPCA(dataf, 2, 'full'))

for i in range(1 ,3):
    print('____________________________________results on train data________________________________________________')
    print("the under data for {} proportion".format(i))
    print()
    under_data = sp.under(dataf, i, y, maj_='no', min_='yes')
    print("------------------------------------------------------------")
    print()
    print("the model classification for {} proportion".format(i))
    print()
    under_Xtrain, under_Xtest, under_ytrain, under_ytest = u.processit(under_data, y, scale='stdscaler', nan=None)
    clf = sp.LogisticRegression()
    # model(clf, under_Xtrain, under_Xtest, under_ytrain, under_ytest)
    print("________________________________________________________________________________________________________")
    # print('____________________________________results on test data________________________________________________')
    data_Xtrain, data_Xtest, data_ytrain, data_ytest = sp.processing(dataf, y)
    sp.model(clf, under_Xtrain, data_Xtest, under_ytrain, data_ytest)

