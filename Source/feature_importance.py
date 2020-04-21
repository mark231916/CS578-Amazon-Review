#!/usr/bin/env python
# coding: utf-8

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import copy
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from random import choices
import os
import pickle
import random


# calculate permutation importance
def permutation_importance(clf, X_test, y_test, feature_indices, n_repeats=5):
    y_test_pred = clf.predict(X_test)
    baseline = accuracy_score(y_test, y_test_pred)  # accuracy score in the original dataset
    imp = []
    for f_idx in feature_indices: # because there're too many words... only compute for the specified word
        scores = []
        for i in range(n_repeats):
            X_test_temp = copy.deepcopy(X_test)
            # shuffle values for the current feature
            X_test_temp[:, f_idx] = np.random.permutation(X_test_temp[:, f_idx])
            y_test_pred = clf.predict(X_test_temp)
            # feature importance score is the baseline accuracy minus accuracy of the shuffled dataset
            scores.append(abs(baseline - accuracy_score(y_test, y_test_pred)))
        #print(feature_names[f_idx], scores)
        imp.append(scores)
    return np.array(imp)


# import dataset
print('------ import dataset ------')
filepath = os.path.join('../dataset/X.csv')
X = pd.read_csv(filepath, index_col=None)
feature_names = X.columns.values
X = X.to_numpy()
filepath = os.path.join('../dataset/y.csv')
y = pd.read_csv(filepath, index_col=None)
y = np.ravel(y.to_numpy())
# split into training set and test set
print('------ split into training set and test set ------')
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2)


print('------ svm ------')
# load model
clf = pickle.load(open('../dataset/svm.pkl', 'rb'))
# permutation feature importance
feature_indices = (-X.sum(axis=0)).argsort()[: int(X.shape[1] * 0.01)]  # only compute for the top 1%-frequent word
feature_imp = permutation_importance(clf, X_test, y_test, feature_indices, n_repeats=10)


sorted_idx = (-feature_imp.mean(axis=1)).argsort()  # top 10 important feature
fig, ax = plt.subplots(figsize=(20, 10))
ax.boxplot(feature_imp[sorted_idx[:20], :].T,
           labels=feature_names[feature_indices[sorted_idx[:20]]])
ax.set_title("Top 20 Permutation Importances (test set) - SVM")
fig.tight_layout()
fig.savefig('permutation_importance_svm.png')

df_svm_perm = pd.DataFrame()
df_svm_perm['word'] = feature_names[feature_indices[sorted_idx]]
df_svm_perm['score'] = feature_imp.mean(axis=1)[sorted_idx]
df_svm_perm.to_csv('perm_importance_svm.csv', index=False)

sorted_idx = (-abs(clf.coef_[0])).argsort()
df_svm_builtin = pd.DataFrame()
df_svm_builtin['word'] = feature_names[sorted_idx]
df_svm_builtin['score'] = clf.coef_[0][sorted_idx]
df_svm_builtin.to_csv('built-in_importance_svm.csv', index=False)


print('------ nn ------')
clf = pickle.load(open('../dataset/nn.pkl', 'rb'))
feature_indices = (-X.sum(axis=0)).argsort()[: int(X.shape[1] * 0.01)]  # only compute for the top 1%-frequent word
feature_imp = permutation_importance(clf, X_test, y_test, feature_indices, n_repeats=10)


sorted_idx = (-feature_imp.mean(axis=1)).argsort()  # top 10 important feature

fig, ax = plt.subplots(figsize=(20, 10))
ax.boxplot(feature_imp[sorted_idx[:20], :].T,
           labels=feature_names[feature_indices[sorted_idx[:20]]])
ax.set_title("Top 20 Permutation Importances (test set) - Neural Network")
fig.tight_layout()
fig.savefig('permutation_importance_nn.png')

df_nn_perm = pd.DataFrame()
df_nn_perm['word'] = feature_names[feature_indices[sorted_idx]]
df_nn_perm['score'] = feature_imp.mean(axis=1)[sorted_idx]
df_nn_perm.to_csv('perm_importance_nn.csv', index=False)


print('------ lr ------')
clf = pickle.load(open('../dataset/lr.pkl', 'rb'))
feature_indices = (-X.sum(axis=0)).argsort()[: int(X.shape[1] * 0.01)]  # only compute for the top 1%-frequent word
feature_imp = permutation_importance(clf, X_test, y_test, feature_indices, n_repeats=10)


sorted_idx = (-feature_imp.mean(axis=1)).argsort()  # top 10 important feature

fig, ax = plt.subplots(figsize=(20, 10))
ax.boxplot(feature_imp[sorted_idx[:20], :].T,
           labels=feature_names[feature_indices[sorted_idx[:20]]])
ax.set_title("Top 20 Permutation Importances (test set) - Logistic Regression")
fig.tight_layout()
fig.savefig('permutation_importance_lr.png')

df_lr_perm = pd.DataFrame()
df_lr_perm['word'] = feature_names[feature_indices[sorted_idx]]
df_lr_perm['score'] = feature_imp.mean(axis=1)[sorted_idx]
df_lr_perm.to_csv('perm_importance_lr.csv', index=False)

sorted_idx = (-abs(clf.coef_[0])).argsort()
df_lr_builtin = pd.DataFrame()
df_lr_builtin['word'] = feature_names[sorted_idx]
df_lr_builtin['score'] = clf.coef_[0][sorted_idx]
df_lr_builtin.to_csv('built-in_importance_lr.csv', index=False)
