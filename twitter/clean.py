import re, string
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import csv
import nltk
nltk.download('wordnet')
nltk.download('popular')
from nltk.stem import PorterStemmer
porter = PorterStemmer()
from pywsd import disambiguate
from pywsd.similarity import max_similarity as maxsim
import time
import pickle
import  argparse
import os
import pandas as pd


#TODO : stemming / lemmatize / negative to positive

abbreviationpatterns = {

    "'re ": "are",
    "'d ": "would",
    " i ": " I ",
    " b ": " be ",
    "doesn't": "does not",
    "don't": "do not",
    " dont ": " do not ",
    "didn't": "did not",
    "hasn't": "has not",
    "hadn't": "had not",
    "haven't": "have not",
    "won't": "will not",
    "wouldn't": "would not",
    "would've": "would have",
    "can't": "can not",
    "cannot": "can not",
    "couldn't": "could not",
    " gotta ": " got to ",
    " shoulda ": " should have ",
    "isn't": "is not",
    " Im ": " I am ",
    " im ": " I am ",
    " I'm ": " I am ",
    " i'm ": " I am ",     ### TODO match start line
    " Ill ": " I will ",   ######TODO confirm not switching 'ill' too   ### TODO isolate "ill + verb for transforming to 'I'll'
    " I'll ": " I will ",
    " i'll ": " I will ",
    " i've ": " I have ",
    " I've ": " I have ",
    " ive ": " I have ",
    " am i ": " am I ",
    " i was ": " I was ",
    " was i ": " was I ",
    " it's ": " it is ",
    "some1": "someone",
    " u ": " you ",
    " ur ": " your ",
    "at em": "at them",
    " yer ": " you are",
    " she's had": "she has had",   ###TODO :  POS or regex to dinstinguish she's + ing [ she is] from she's + VERB past participle [she has]
    "what 2 do": "what to do",
    " yrs ": " years ",
    "that's": "that is",
    " watcha ": " what are you ",
    " wutcha ": " what are you ",
    " witcha ": " with you "

     }

casualpatterns = {
    "coz": "because",
    "gotta": "got to",
    "nope": "no",
    "yep": "yes",
    '\bu\b': "you"
}

#TODO adapt. Declaring patterns in init(self) makes no sense if the replace method is the same - categorize replacement patterns
class PatternReplace():
    def __init__(self, patterns):
        self.patterns = patterns

    def replace(self, sent):
        for (raw, rep) in self.patterns.items():
            regex = re.compile(raw)
            sent = re.sub(regex, rep, sent)
        return sent


class RepeatReplace():
    def __init__(self):
        self.regex = re.compile(r'(\w*)(\w)\2(\w*)')
        self.repl = r'\1\2\3'

    def replace(self, word):
        if wordnet.synset(word):
            return word
        loop_res = self.regex.sub(self.rep, word)
        if (word == loop_res):
            return loop_res
        else:
            return self.replace(loop_res)


#TODO map wording and sentence to replace in sentence directly

class Desambig():

    def replace(self, sentence):
        wording = []
        wording.append(disambiguate(sentence, algorithm=maxsim, similarity_option='wup', keepLemmas=True))
        for i in wording:
            for j in i:
                if j is not None:
                    j = j.replace(wordnet.synsets(j).name)
        return wording


class WordReplace():
    def __init__(self, word_map):
        self.word_map = word_map

    def replace(self, word):
        return self.word_map.get(word, word)


class AntonymReplace(object):

    def replace(self, word):
        antonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                #print(lemma)
                for antonym in lemma.antonyms():
                    antonyms.add(antonym.name())
                    print(antonyms)
        if len(antonyms) == 1:
            return antonyms.pop()
        else:
            return None


    def rollreplace(self, sentence):
        i = 0
        sent = word_tokenize(sentence)
        words = []
        fsent = ""
        while i < len(sent):
            word = sent[i]
            if word == 'not' and i+1 < len(sent):
                ant = self.replace(sent[i+1])
                if ant:
                    words.append(ant)
                    fsent += ant + " "
                    i += 2
                    continue
            words.append(word)
            fsent += word + " "
            i +=1
        return fsent

class Cleanr():

    @staticmethod
    def multipleReplace(oldString, rxpattern, replacement):
    # Iterate over the strings to be replaced
        for i, elem in enumerate(rxpattern):
        # Check if one of the patterns matches in the string
            oldString = re.sub(elem, replacement, oldString)

        return oldString

    @staticmethod
    def stemSentence(tokenized_string):
        stem_sentence=[]
        for word in tokenized_string:
            stem_sentence.append(porter.stem(word))
            stem_sentence.append(" ")
        return "".join(stem_sentence)


s = 'is it true'
tok = word_tokenize(s)
stemmed = Cleanr.stemSentence(tok)

###############################################################

#                         USAGE                               #

###############################################################

rep = PatternReplace(abbreviationpatterns)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str)
    parser.add_argument('--outfile', type=str)

    args = parser.parse_args()

    path_= args.infile
    df = pd.read_csv(path_, sep=';')

    t3 = time.time()

    pattern0_0 = '([@])([a-z])'
    pattList = [re.compile('((http)(.*?)[?=\s])'), re.compile('((http)(.*?)$)'), re.compile('([#]+[A-Za-z]{2,})'), re.compile('([@])'), re.compile('(^\s+[-]\s+)'), re.compile('(^\s+)',)]
    def to_upper_user(match):
        return match.group(2).upper()
    rep = PatternReplace(abbreviationpatterns)
    exclude = set(string.punctuation)
    table = str.maketrans("", "", string.punctuation)
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    df['tweet'] = df['tweet'].apply(lambda x: re.sub(pattern0_0, to_upper_user, x))
    df['tweet'] = df['tweet'].apply(lambda x: Cleanr.multipleReplace(x, pattList, ""))
    df['tweet'] = df['tweet'].apply(lambda x: rep.replace(x))
    df['tweet'] = df['tweet'].apply(lambda x: x.translate(table))


    t4 = time.time()
    print(t4-t3)

    outpath_ = args.outfile
    df.to_csv(outpath_, sep = ';', index=False)
