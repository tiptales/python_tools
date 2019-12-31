
import tweepy
import pandas as pd
import time
import os
import argparse
import json

consumer_key=''
consumer_secret=''
acces_token=''
acces_secret=''

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(acces_token, acces_secret)
api = tweepy.API(auth)

def scrap_(tweetID, path_):
    """
    :param tweetID:
    :param path_:
    :var los :list of ID sent to twitter
    :var lop : list of ID actually returned
    :return: los, lop and write intermediate scrapped tweets to disk
    """
    los = []
    lop = []
    files = [i for i in tweetID]
    for i in range(0, len(files), 10):
        batch = files[i:i + 10]
        print(batch)
        time.sleep(1)
        bidu = []
        tweets = api.statuses_lookup(batch)
        print(i)
        los.append(files[i:i+10])
        #print(los)
        los_ = [item for sub in los for item in sub]
        with open('los.txt', 'a') as f:
            f.write(str(los_) + '\n')
        for tweet in tweets:
            t = tweet.text
            id = tweet.id
            bidu.append([t, id])
            lop.append(id)

        df = pd.DataFrame(bidu)

        if not os.path.exists(path_):
            os.makedirs(path_)
        df.to_csv(os.path.join(path_, (str(i) + '_tweets.csv')), sep=';', index=False, header=0)

    return los, lop



#Rewrite the scrap function to take into account the IDS already parsed, the IDs having actual results
# and excluding both from new scrapping

# Make a while loop to call this scrap fonction while the comparison between reference ID set and the set of IDS still to parse is not empty

if __name__ == '__main__':

    parser=argparse.ArgumentParser()
    parser.add_argument('--init')
    parser.add_argument('--augment', type=int, help='augment finance directory from count')
    parser.add_argument('--outrepo', type=str, help='output repository path')
    args=parser.parse_args()

    if args.init:

    #### TODO this part is going to be called if argparse argument init is called
        tweetdf = pd.read_csv('BIDU.txt', sep='\t')
        tweetdf = tweetdf[['TweetID', 'Label']]
        tweetdf = tweetdf.sort_values('TweetID')
        tweetID = tweetdf['TweetID']
        #tweetID = list(tweetID[0:100])
        los_, lop_ = scrap_(tweetID, os.path.join('finance', 'finance0'))


    elif args.augment:

        with open('los.txt', 'r') as f:
            los_ = f.readlines()

        tweetdf = pd.read_csv('BIDU.txt', sep='\t')
        tweetdf = tweetdf[['TweetID', 'Label']]
        tweetdf = tweetdf.sort_values('TweetID')

        tweetID = tweetdf['TweetID']
        lot = list(tweetID)
        setdifference = set(los_) ^ set(lot)
        print(len(lot), len(setdifference), len(set(los_)), len(los_))
        time.sleep(1)

        count = args.augment
        setdiff_ = set(los_) ^ set(lot)
        if setdiff_ is not None:
            los_, lop_= scrap_(list(setdiff_), os.path.join('finance', ('finance'+str(count))))
            print(len(lot), len(setdiff_), len(los_))


        #
        #
        #
        # count = args.augment
        # while True:
        #     setdiff_ = set(los_) ^ set(lot)
        #     if setdiff_ is not None:
        #         los_, lop_= scrap_(list(setdiff_), os.path.join('finance', ('finance'+str(count))))
        #         count=+1
        #         print(len(lot), len(setdiff_), len(los_))


