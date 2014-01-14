#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
:Name:
    classify.py

:Authors:
    Soufian Salim (soufi@nsal.im)

:Date:
    12 january 2014 (creation)

:Description:
    Text classifier
"""
import warnings
import sys
import getopt

from nltk.corpus import reuters
from nltk.classify.scikitlearn import SklearnClassifier

from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfTransformer

# Global vars
c = f = False  # Options

# EVIL!
warnings.filterwarnings("ignore")


# Main
def main(argv):
    ###################### GETOPT ######################

    global c, f

    optErrorMsg = "Usage: classify.py -c [NB | SVM | DT | KNN] -f [tfidf | chi2]"

    try:
        opts, args = getopt.getopt(argv, "c:f:")
    except getopt.GetoptError:
        sys.exit(optErrorMsg)
    for opt, arg in opts:
        if opt == "-c":
            c = arg
        elif opt == "-f":
            f = arg

    if not f in ["tfidf", "chi2"] or not c in ["NB", "SVM", "DT", "KNN"]:
        sys.exit(optErrorMsg)

    ####################### SETS #######################

    train = [(features(reuters.words(fileid)), category)
        for category in reuters.categories()
        for fileid in reuters.fileids(category) if fileid.startswith("train")]

    test = [(features(reuters.words(fileid)), category)
        for category in reuters.categories()
        for fileid in reuters.fileids(category) if fileid.startswith("test")]

    ##################### PIPELINE #####################

    plist = []

    # Feature selection method
    if f == "tfidf":
        plist.append(("tfidf", TfidfTransformer()))
    elif f == "chi2":
        plist.append(("chi2", SelectKBest(chi2, k=250)))
    else:
        sys.exit(optErrorMsg)

    # Classification method
    if c == "NB":
        plist.append(("NB", MultinomialNB()))
    elif c == "SVM":
        plist.append(("SVM", LinearSVC()))
    elif c == "DT":
        plist.append(("DT", DecisionTreeClassifier()))
    elif c == "KNN":
        plist.append(("KNN", KNeighborsClassifier()))
    else:
        sys.exit(optErrorMsg)

    pipeline = Pipeline(plist)
    classifier = SklearnClassifier(pipeline)
    classifier.train(train)

    #################### EVALUATION ####################

    data = []
    gold = []

    for f in test:
        data.append(f[0])
        gold.append(f[1])

    guess = classifier.batch_classify(data)

    precision = metrics.precision_score(gold, guess)
    recall = metrics.recall_score(gold, guess)

    print "Precision:", precision
    print "Recall:", recall
    print "F-measure:", (2.0 * (recall * precision)) / (recall + precision)


def features(words):
    return dict([(word, True) for word in words])

# Launch
if __name__ == "__main__":
    main(sys.argv[1:])
