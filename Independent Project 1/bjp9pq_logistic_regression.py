from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import string
from sklearn.linear_model import Perceptron
from sklearn.model_selection import KFold
import os
import numpy as np


def log_reg(xtrain_path, ytrain_path, xtest_path):
    xdata = []
    with open(xtrain_path, 'r') as f:
        for row in f:
            xdata.append(row)
    xdata = np.asarray(xdata)
    ydata = []
    with open(ytrain_path, 'r') as f:
        for row in f:
            ydata.append(int(row.strip('\n')))
    ydata = np.asarray(ydata)

    kf = KFold(n_splits=2)
    d =0

    testdata = []
    with open(xtest_path, 'r') as f:
        for row in f:
            testdata.append(row)
    testdata = np.asarray(testdata)

    for train_index, test_index in kf.split(xdata):
        if d == 1: break
        d +=1
        X_train, X_test = xdata[train_index], xdata[test_index]
        y_train, y_test = ydata[train_index], ydata[test_index]
        count_vect = CountVectorizer(ngram_range=(1,2))
        xbow = count_vect.fit_transform(X_train)
        xtestbow = count_vect.transform(X_test)
        scores = []
        clf = LogisticRegression()
        #clf = Perceptron()
        clf.fit(xbow, y_train)
        score = clf.score(xtestbow, y_test)

        bowtesting = count_vect.transform(testdata)
        output = clf.predict(bowtesting)
        with open('bjp9pq-lr-test.pred','w+') as f:
            for line in output:
                f.write(str(line) + '\n')



    pass

if __name__ == '__main__':
    trainx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trn.data')
    trainy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trn.label')
    devx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dev.data')
    devy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dev.label')
    test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tst.data')
    log_reg(trainx_path, trainy_path, test_path)
    print('___________________________')
    #log_reg(devx_path, devy_path, test_path)