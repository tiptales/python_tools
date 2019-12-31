import pandas as pd
from scipy import stats
import numpy as np
from itertools import chain, combinations
from collections import defaultdict
import pingouin as pg
from sklearn.decomposition import PCA
import argparse

##########################################################################
#                                                                        #
#                          NON PARAMETRIC TESTS                          #
#                                                                        #
##########################################################################

##########################################################################
#                                                                        #
#                          KRUSKAL WALLIS                                #
#               HO : all medians are equal                               #
#               H1 : at least two medians are different                  #
#                                                                        #
##########################################################################
##########################################################################
#                                                                        #
#                          PAIRWISE WILCOX                               #
#               HO : med1 == med2                                        #
#               H1 : med1 6= med2 (”two.sided”), med1 < med2 (”less”)    #
#                    or med1 > med2 (”greater”)                          #
#               A typical rule is to require that n > 20                 #
#                                                                        #
##########################################################################

def pairwisetest(df, y):
    """
    :param df: dataframe. Only qualitative variables will be tested against response variable
    :param y: categorical response variable
    :return: test stat and p-values for each y-group for all quantitative variables
    """
    label, idx = np.unique(df[str(y)], return_inverse=True)
    kruskal_ = defaultdict(list)
    mannwhitneyu_ = defaultdict(list)
    numlist = df.select_dtypes(exclude=['object']).columns.tolist()
    for name in numlist:
        x = np.array(df[str(name)])
        groups = [x[idx == i] for i, l in enumerate(label)]
        H, pk = stats.kruskal(*groups)
        kruskal_[name].append((H, pk))
        kk = pd.DataFrame([(k_, v[0][0], v[0][1]) for k_, v in kruskal_.items()], columns=['Variable', 'Kruskal_Stat', 'Kruskal_p-value'])
        Z, pw = stats.mannwhitneyu(*groups)
        mannwhitneyu_[name].append((Z, pw))
        mw = pd.DataFrame([(k_, v[0][0], v[0][1]) for k_, v in mannwhitneyu_.items()], columns=['Variable', 'MannWU_Stat', 'MannWU_p-value'])
        pt = pd.merge(kk, mw, on='Variable')

    return pt


##########################################################################
#                                                                        #
#                          Chi Square Test                               #
#               HO : factor 1 and factor 2 are independent              #
#               H1 : at least two medians are different                  #
#                                                                        #
##########################################################################

def chi2(df):
    """
    :param df:
    :return: Table of chi square test for each combination of quantitative variables
    """
    chidict = defaultdict(list)
    objectlist = df.select_dtypes(include=['object']).columns.tolist()
    tl = list(chain.from_iterable(combinations(objectlist, r) for r in range(len(objectlist) + 1)))
    uniquetuple = [i for i in tl if len(i) == 2 and i[0] != i[1]]

    ctbl1 = []
    ctbl2 = []
    for i, names in enumerate(uniquetuple):

        ctbl1.append(names)
        try:
            contingency_table = pd.crosstab(df[names[0]], df[names[1]], margins=True)
            f_obs = np.array([contingency_table.iloc[0][0:contingency_table.shape[1]].values,
                              contingency_table.iloc[1][0:contingency_table.shape[1]].values])
            chidict[i] = (names[0], names[1], stats.chi2_contingency(f_obs)[0:3])
        except:
            ctbl2.append(names)
    if len(ctbl2) !=0:
        print((len(ctbl1) - len(ctbl2)), ' scores have not been computed, check rejected variables')
        print(ctbl2)

    chidf = pd.DataFrame([(v[0], v[1], v[2][0], v[2][2], v[2][1]) for k_, v in chidict.items()],
                         columns=['Variable_1', 'Variable_2', 'Chi_Stat', 'Chi_df', 'Chi_p-value'])

    return chidf

##########################################################################
#                                                                        #
#                            PARAMETRIC TESTS                            #
#                                                                        #
##########################################################################
##########################################################################
#                                                                        #
#                          Correlation Coefficients                      #
#                                                                        #
##########################################################################

def corrcoef(df, method):
    """
    :param df: dataframe
    :param method: a method from the pandas.corr() methods
    :return: table of sorted correlation coefficients for quantitative variables
    """

    numlist = df.select_dtypes(exclude=['object']).columns.tolist()
    cormat = df[numlist].corr(method= method).abs()
    sortedcoef = (cormat.where(np.triu(np.ones(cormat.shape), k=1).astype(np.bool))
           .stack()
           .sort_values(ascending=False))
    return sortedcoef


##########################################################################
#                                                                        #
#                          ANOVA TEST                                    #
#    All quantitative variables are tested against the binary response   #
#                                                                        #
##########################################################################


def aovtables(df, y1, y2):
    """
    :param df: dataframe
    :param y1: factor 1
    :param y2: factor 2
    :return: Anova tables (III) testing the influence of y1 and y2 factors on each quantitative variable
    """
    aovlist = []
    numlist = df.select_dtypes(exclude=['object']).columns.tolist()
    for name in numlist:
        aovlist.append([name, pg.anova(dv=name, between=[str(y1), str(y2)], data=df, detailed=True)])

    return aovlist

##########################################################################
#                                                                        #
#                          PCA TESTS                                     #
#                                                                        #
##########################################################################


def XDPCA(df, n_components, svd_solver):
    """
    :param df: dataframe, only quantitative values are considered, are supposed scaled
    :param n_components: int, nb of components
    :param svd_solver: solver of scikit PCA
    :return: PCA object
    """

    pca = PCA(n_components=n_components, svd_solver=svd_solver)
    numlist = df.select_dtypes(exclude=['object']).columns.tolist()
    pca_ = pca.fit(df[numlist])

    maxcomp = [np.abs(pca_.components_[i]).argmax() for i in range(pca_.components_.shape[0])]
    fnames = df[numlist].columns.tolist()
    maxcomp_names = [fnames[maxcomp[i]] for i in range(pca_.components_.shape[0])]
    dict_ = {'PC{}'.format(i + 1): maxcomp_names[i] for i in range(pca_.components_.shape[0])}
    dfpca = pd.DataFrame(sorted(dict_.items()))
    dfpca['Explained Variance'] = pca_.explained_variance_ratio_

    return dfpca


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, help='input file path')
    parser.add_argument('--sep', type=str, help='pandas sep if needed')

    parser.add_argument('--Y', type=str, help='value of interest')
    parser.add_argument('--method', type=str, help='method for correlation coef computation')
    parser.add_argument('--Y2', type=str, help='second factor for ANOVA tables')
    parser.add_argument('--dim', type=int, help ='number of dimensions for PCA analysis')

    args = parser.parse_args()
    dataf = pd.read_csv(args.infile, sep=args.sep)
    Y, method, Y2, dim = args.Y, args.method, args.Y2, args.dim

    print(dataf.dtypes)
    print('__________________________________________________________________________')
    print(pairwisetest(dataf, Y))
    print('__________________________________________________________________________')
    print(chi2(dataf))
    print('__________________________________________________________________________')
    print(corrcoef(dataf, method=method))
    print('__________________________________________________________________________')
    [print(i, '\n') for i in aovtables(dataf, Y2, Y)]
    print('__________________________________________________________________________')
    print(XDPCA(dataf, dim, 'full'))

