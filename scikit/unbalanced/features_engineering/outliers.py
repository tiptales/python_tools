import numpy as np
from scipy import stats



def cutIQR(df):

    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    print(IQR)
    df_ = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
    print(df.shape, df_.shape)

    return df_



def cutZ(df, z):
    """
    :param df: dataframe
    :param z: max zscore (recommanded : 3)
    :return: reduced dataframe
    """
    numlist = df.select_dtypes(exclude=['object']).columns.tolist()
    zs = np.abs(stats.zscore(df[numlist]))
    print(zs)
    df_ = df[(zs < z).all(axis=1)]

    return df_


def lower_bound(tmp, limit):
    """
    :param tmp: df
    :param limit: lower_limit will be std*limit
    :return:
    """
    # Set upper and lower limit to 5* standard deviation
    tmp_std = np.std(tmp)
    tmp_mean = np.mean(tmp)
    cut_off = tmp_std * limit

    lower_limit = tmp_mean - cut_off
    print(lower_limit)

    return lower_limit

def upper_bound(tmp, limit):
    """
    :param tmp: df
    :param limit: limit ratio. upper_limit will be std*limit
    :return:
    """
    tmp_std = np.std(tmp)
    tmp_mean = np.mean(tmp)
    cut_off = tmp_std * limit

    upper_limit = tmp_mean + cut_off
    print(upper_limit)

    return upper_limit



def cutNA(df, level):
    """
    :param df: dataframe
    :param level: int: threshold of missing data over which the variable is rejected
    :return:
    """
    nans = df.apply(lambda x: sum(x.isna()), axis = 0)
    lofnans = nans[nans < level].index.to_list()
    df_ = df[lofnans]

    return df_