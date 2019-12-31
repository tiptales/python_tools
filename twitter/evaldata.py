
import pickle
import pandas as pd
import csv
from glob import glob
import os
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('--inID', default='BIDU.txt', type=str, help='input file of all IDS')
    parser.add_argument('--inrepo', type=str, help='repo of parsed IDS')
    # parser.add_argument('--colnames', type=list, help='file headers')
    parser.add_argument('--outfile', type=str, help='output file')
    args = parser.parse_args()

    inpath_, outpath_ = args.inrepo, args.outfile
    ttrain = []
    ldlf = []
    for filename in glob(os.path.join(inpath_+'/**/*.csv'), recursive=True):
        print(filename)
        with open(filename, 'r') as f:
            tmp1 = csv.reader(f)
            #next(tmp1, None)    ### comment this line if headers have not been saved previously
            for i in tmp1:
                ttrain.append(i)


            try:
                dfl = pd.read_csv(filename, sep=';')
                dfl.columns = ['tweet', 'TweetID']
                ldlf.append(dfl)
            except:
                pass
    dfall = pd.concat(ldlf)

    print(len(dfall['TweetID'].unique()))
    dfall_ = dfall.drop_duplicates(subset='TweetID')
    tweetdf = pd.read_csv('BIDU.txt', sep='\t')
    df = pd.merge(dfall_, tweetdf, on='TweetID')
    df.to_csv(outpath_, sep = ';', index=False)

