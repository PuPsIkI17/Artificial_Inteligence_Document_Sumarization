from os import listdir
from os.path import isfile, join
import random
from random import randint
from math import log
import nltk
import re
import sys

#STOP_WORDS = []
STOP_WORDS = [line.strip() for line in open("../sumarizarea-documentelor/stop_words")]
stemmer = nltk.PorterStemmer()
withoutStopWords = 0
lemantization = 0

def naiveBayes():
    s = random.getstate()
    dirPath = "../sumarizarea-documentelor/BBC News Summary/News Articles";
    dirSolPath = "../sumarizarea-documentelor/BBC News Summary/Summaries";
    dirs = [f for f in listdir(dirPath)]
    trainingCls = {}
    testCls = {}
    for dir in dirs:
        filesPath = dirPath + "/" + dir;
        filesSolPath = dirSolPath + "/" + dir;
        files = [f for f in listdir(filesPath) if isfile(join(filesPath, f))]
        trainingFiles = [];
        testFiles = [];
        for file in files:
            rndValue = randint(0, 100)
            if rndValue < 80:
                trainingFiles += [file];
            else:
                testFiles += [file];
        trainingCls[filesPath] = (trainingFiles, filesSolPath)
        testCls[filesPath] = (testFiles, filesSolPath)
    random.setstate(s)
    return (trainingCls, testCls)

def parse_document(path, file):
    path = path + '/' + file;
    f = open(path, "r")
    for word in re.findall(r"[-\w']+", f.read()):
        if withoutStopWords == 0:
            if len(word) > 1 and word not in STOP_WORDS:
                yield word
        else:
            if len(word) > 1:
                yield word

def count_words(path, files):
    vocabulary = {}
    words_no = 0
    for file in files:
        text = parse_document(path, file)
        for k in text:
            if lemantization == 0:
                k = stemmer.stem(k)
            if k in vocabulary:
                vocabulary[k] += 1
            else:
                vocabulary[k] = 1
            words_no += 1
    return (vocabulary, words_no)

def predict(params, path, file, alpha=1):
    mn = -(sys.maxsize-1)
    answer = ""
    for cls in params:
        pos_cls = log(0.2)
        (vocabulary, words_no) = params[cls]
        tokens = parse_document(path, file)
        for k in tokens:
            pos = 0.0
            if lemantization == 0:
                k = stemmer.stem(k)
            if k in vocabulary:
                pos = vocabulary[k]
            probab = (pos + alpha) / (words_no + len(vocabulary) * alpha)
            pos_cls += log(probab)
        if pos_cls > mn:
            mn = pos_cls
            answer = cls
    return answer

def classification(StopWords, lmnt):
    global withoutStopWords, lemantization
    withoutStopWords = StopWords
    lemantization = lmnt
    (trainingCls, testCls) = naiveBayes()

    cls = {}
    for path in trainingCls:
        (value, solPath) = trainingCls[path]
        ans = count_words(path, value)
        cls[path] = ans

    goodClassification = 0
    badClassification = 0
    for path in testCls:
        (files, solPath) = testCls[path]
        for file in files:
            pred = predict(cls, path, file)
            if pred == path:
                goodClassification += 1;
            else:
                badClassification += 1;
    print("Total clasificari corecte: " + str(goodClassification) + " si clasificari gresite:" + str(badClassification))
    prct = goodClassification/(goodClassification+badClassification)
    print("Corectitudinea: " + str(prct))


if __name__ == '__main__':
    print("Fara lemantizare si fara stop words:")
    classification(1, 1)
    print("Folosind stop words:")
    classification(0, 1)
    print("Folosind stop words si lemantizare:")
    classification(0, 0)
