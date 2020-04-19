#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 22:55:45 2020

@author: yangmian
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import collections
from itertools import product
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import LinearSVC, SVC
import math
import pandas as pd

def Tune_batchsize(samplesize,clf,X,y):
    Accuracy,sample=[],[]
    for k in samplesize:
        n = len(y) # number of training samples
        val_size = math.floor(n / k)
        accu =0
        for round in range(k):
            if round == k - 1:
                indices = range(round * val_size, n)
            else:
                indices = range(round * val_size, (round + 1) * val_size)
            X_val = X[indices]
            y_val = y[indices]
            X_train = np.delete(X, indices, axis=0)
            y_train = np.delete(y, indices, axis=0)
            clf.fit(X_train,y_train)
            pred=clf.predict(X_val)
            accu+=sum(pred==y_val.flatten())/len(pred)
        Accuracy+=[accu/k]
        sample+=[k]
    plt.plot(sample,Accuracy)
    plt.xlabel('k fold')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Different Number of K value') 
    plt.show()


filepath = "dataset/X.csv"
X = pd.read_csv(filepath, index_col=None)
X = X.to_numpy()
filepath = "dataset/y.csv"
y = pd.read_csv(filepath, index_col=None)
y = np.ravel(y.to_numpy())
p = np.random.permutation(len(y))
X = X[p]
y = y[p]
samplesize=range(5,10)
clf=SVC(random_state=0,kernel='rbf',C=1.0)
Tune_batchsize(samplesize,clf,X,y)        

    
def Tune_hyperparameter(clf,params,X,y,k,thres):
    if thres:
        thres_grid=params['thres']
        params={keys:v for keys,v in svm_params.items() if keys!='thres'}
    items = sorted(params.items())
    keys, values = zip(*items)
    grid = []
    for v in product(*values):
        grid.append(dict(zip(keys, v)))    
    # iterate over parameter_grid
    params_history =collections.defaultdict(list)
    for idx, param in enumerate(grid):
        print('running {}/{} in parameter grid ...'.format(idx + 1, len(grid)))
        clf.set_params(**param) # set the classifier's hyperparameter to the current parameter
        n = len(y) # number of training samples
        val_size = math.floor(n / k)
        for round in range(k):
            if round == k - 1:
                indices = range(round * val_size, n)
            else:
                indices = range(round * val_size, (round + 1) * val_size)
            X_val = X[indices]
            y_val = y[indices]
            X_train = np.delete(X, indices, axis=0)
            y_train = np.delete(y, indices, axis=0)
        clf.fit(X_train,y_train)
        if not thres:
            pred=clf.predict(X_val)
            pred=np.array(pred).reshape((-1,1))
            params_history[keys[0]]+=[param.get(keys[0])]
            params_history[keys[1]]+=[param.get(keys[1])]
            params_history['Accuracy']+=[sum(pred==y_val)/len(y_val)]
        else:
            for threshold in thres_grid:
                pred=(clf.predict_proba(X_val)[:,0]>threshold)*1
                params_history[keys[0]]+=[param.get(keys[0])]
                params_history['Threshold']+=[threshold]
                params_history['Accuracy']+=[sum(pred==y_val.flatten())/len(y_val)]
    return params_history
##Tuning of logistic regression
lg_params={'penalty':['l1','l2', 'elasticnet'],'C':np.logspace(-3,-0.5,20)}
lg_history=Tune_hyperparameter(LogisticRegression(solver='saga',l1_ratio=0.5),lg_params,X,y,5,thres=False)

## Tuning of SVM model
svm_params={'kernel':['linear', 'rbf'],'thres':np.linspace(0.2,0.7,30)}
svm_history=Tune_hyperparameter(SVC(C=1.0,probability=True),svm_params,X,y,5,thres=True)

def plotting(x,y,z,xlabel,ylabel,title):
    Z=z.reshape((len(y),len(x)))
    X, Y = np.meshgrid(x, y)
    ax = plt.axes(projection='3d')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel('Accuracy');
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,cmap='viridis');
    ax.set_title(title);
##Plot the tuning for logistic regression   
lg_x=range(len(lg_params['C']));lg_y=range(len(lg_params['penalty']));lg_z=np.array(lg_history['Accuracy'])
lg_xlabel='C';lg_ylabel='Penalty';lg_title='Accuracy vs Hyperparameters Tuning for Logistic Regression'
plotting(lg_x,lg_y,lg_z,lg_xlabel,lg_ylabel,lg_title)
## plot the tuning for SVM 
svm_x=range(len(svm_params['kernel']));svm_y=range(len(svm_params['thres']));svm_z=np.array(svm_history['Accuracy'])
svm_xlabel='Kernel';svm_ylabel='Threshold';svm_title='Accuracy vs Hyperparameters Tuning for SVM'
plotting(svm_x,svm_y,svm_z,svm_xlabel,svm_ylabel,svm_title)
