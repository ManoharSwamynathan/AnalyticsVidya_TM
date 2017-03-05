from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import cross_validation

import os
os.chdir('C:\\AV_TM')
train = pd.read_csv('train_MLWARE1.csv') 
test = pd.read_csv('test_MLWARE1.csv')

y_train = train.label

vectorizer = TfidfVectorizer(sublinear_tf=True, smooth_idf = True, max_df=0.5,  ngram_range=(1, 2), stop_words='english')
X_train = vectorizer.fit_transform(train.tweet)
X_test = vectorizer.transform(test.tweet)
y_train = train.label

print("Train Dataset")
print("n_samples: %d, n_features: %d" % X_train.shape)

print("Test Dataset")
print("n_samples: %d, n_features: %d" % X_test.shape)


from sklearn.ensemble import RandomForestClassifier
num_trees = 50

clf = RandomForestClassifier(n_estimators=num_trees, n_jobs=-1).fit(X_train, y_train)

y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

print 'Train accuracy_score: ', metrics.accuracy_score(y_train, y_train_pred)

test['label'] = y_test_pred
del test['tweet']

test.to_csv('SampleSubmission_MLWARE1.csv')

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression().fit(X_train, y_train)

y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

print 'Train accuracy_score: ', metrics.accuracy_score(y_train, y_train_pred)

test['label'] = y_test_pred
del test['tweet']

test.to_csv('SampleSubmission_MLWARE1.csv')
