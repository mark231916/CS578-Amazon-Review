from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier
from itertools import product
import numpy as np
import math
import pandas as pd
from random import choices
import os
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import collections
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def import_data(filepath):
    data = pd.read_csv(filepath)
    y = data['label']
    X = data.drop(columns=['label'])
    # shuffle X and y 
    p = np.random.permutation(len(y))
    X = X[p]
    y = y[p]
    return X, y


def get_SVC(C=1.0, kernel="linear"):
    return SVC(random_state=0,
               kernel=kernel,
               probability=True,
               C=C)

def get_accuracy(y_test, y_pred):
    temp = y_test[y_test == y_pred]
    return len(temp) / len(y_test)


def train_classifier(X_train, y_train, clf):
    # input:
    #     X_train, y_train: training set
    #     clf: a sklearn classifier, SVM or NN or LR
    clf.fit(X_train, y_train)


def test_classifier(X_test, y_test, clf, metrics='accuracy'):
    # input:
    #     X_test, y_test: test (or validation) set
    #     clf: a trained classifier, SVM or NN or LR
    #     metrics: 'accuracy' or 'precision' or 'recall' or 'roc_auc' or ...
    # output:
    #     score: performance score on the test set
    y_pred = clf.predict(X_test)
    return get_accuracy(y_test, y_pred)


def kfold_score(X_train_val, y_train_val, clf, k, metric='accuracy'):
    n = len(y_train_val) # number of training samples
    val_size = math.floor(n / k)
    cv_scores = []
    for round in range(k):
        if round == k - 1:
            indices = range(round * val_size, n)
        else:
            indices = range(round * val_size, (round + 1) * val_size)
        X_val = X_train_val[indices]
        y_val = y_train_val[indices]
        X_train = np.delete(X_train_val, indices, axis=0)
        y_train = np.delete(y_train_val, indices, axis=0)
        train_classifier(X_train, y_train, clf)
        cv_scores.append(test_classifier(X_val, y_val, clf, metric))
    return sum(cv_scores) / k


def bootstrap_score(X_train_val, y_train_val, clf, B=5, metric='accuracy'):
    n = len(y_train_val) # number of training samples
    bs_scores = [] # bootstrap scores
    for round in range(B):
        indices = choices(range(n), k=n) # pick n samples with replacement
        #indices = random.sample(range(n), n)
        X_train = X_train_val[indices]
        y_train = y_train_val[indices]
        X_val = np.delete(X_train_val, list(set(indices)), axis=0)
        y_val = np.delete(y_train_val, list(set(indices)), axis=0)
        train_classifier(X_train, y_train, clf)
        bs_scores.append(test_classifier(X_val, y_val, clf, metric))
    return sum(bs_scores) / B # return the average score


def cross_validation_train(X_train_val, y_train_val, clf, parameter_grid, cv_technique='k-fold', k=5, B=5, metric='accuracy'):
    # computes the cartesian product of parameter_grid
    items = sorted(parameter_grid.items())
    keys, values = zip(*items)
    grid = []
    for v in product(*values):
        grid.append(dict(zip(keys, v)))
    # iterate over parameter_grid
    params, scores, best_score, best_params = [], [], -1, None
    params_history =collections.defaultdict(list)
    for idx, param in enumerate(grid):
        print('running {}/{} in parameter grid ...'.format(idx + 1, len(grid)))
        if (isinstance(clf, SVC)):
            print('C = {}, kernel = {}'.format(param.get('C'), param.get('kernel')))
        elif (isinstance(clf,MLPClassifier)):
            print('num of hidden layer = {}, num of neuron = {}, solver = {}'.format(len(param.get('hidden_layer_sizes')), 
                    param.get('hidden_layer_sizes')[0], param.get('solver')))
        clf.set_params(**param) # set the classifier's hyperparameter to the current parameter
        # if technique='k-fold', then use k-fold cross validation
        if (cv_technique == 'k-fold'):
            score = kfold_score(X_train_val, y_train_val, clf, 5, metric=metric)
        elif (cv_technique == 'bootstrap'):
            score = bootstrap_score(X_train_val, y_train_val, clf, B=B, metric=metric)
        print('cv score: {}'.format(score))
        params.append(param)
        scores.append(score)
        params_history['Num of layers']+=[len(param.get('hidden_layer_sizes'))]
        params_history['Num of neurons']+=[param.get('hidden_layer_sizes')[0]]
        params_history['Accuracy']+=[score]
        if (score > best_score):
            best_score = score
            best_param = param
    clf.set_params(**best_param)
    train_classifier(X_train_val, y_train_val, clf)
    return clf, params, scores, best_param, best_score, params_history

def kfold_withthres(X_train_val, y_train_val, clf, k,threshold):
    n = len(y_train_val) # number of training samples
    val_size = math.floor(n / k)
    cv_scores = []
    for round in range(k):
        if round == k - 1:
            indices = range(round * val_size, n)
        else:
            indices = range(round * val_size, (round + 1) * val_size)
        X_val = X_train_val[indices]
        y_val = y_train_val[indices]
        X_train = np.delete(X_train_val, indices, axis=0)
        y_train = np.delete(y_train_val, indices, axis=0)
        clf.fit(X_train,y_train)
        pred=(clf.predict_proba(X_val)[:,1]>threshold)*1
        cv_scores.append(get_accuracy(y_val,pred))
    return sum(cv_scores) / k


def Tune_batchsize(size,clf,X,y):
    error_his,size_his=[],[]
    for i,k in enumerate(size):
        print('running {}/{} in batchsize grid ...'.format(i+1, len(size)))
        accuracy=kfold_score(X, y, clf, k, metric='accuracy')
        error_his+=[1-accuracy]
        size_his+=[k]
    plt.plot(size_his,error_his)
    plt.xlabel('k fold')
    plt.ylabel('Error of Validation set')
    plt.title('Error vs Different Number of K choice') 
    plt.savefig('Error vs Different Number of K choice.png')

 
def Tune_hyperparameter(clf,params,X,y,k,thres):
    if thres:
        thres_grid=params['thres']
        params={keys:v for keys,v in params.items() if keys!='thres'}
    items = sorted(params.items())
    keys, values = zip(*items)
    grid = []
    for v in product(*values):
        grid.append(dict(zip(keys, v)))    
    # iterate over parameter_grid
    params_history =collections.defaultdict(list)
    for idx, param in enumerate(grid):
        print(idx,param)
        print('running {}/{} in parameter grid ...'.format(idx + 1, len(grid)))
        clf.set_params(**param) # set the classifier's hyperparameter to the current parameter
        n = len(y) # number of training samples
        val_size = math.floor(n / k)
        if thres: # tune the threshold 
            for i, threshold in enumerate(thres_grid):
                print('running {}/{} in thres_grid ...'.format(i+1, len(thres_grid)))
                accuracy=kfold_withthres(X, y, clf, k,threshold)
                params_history['Threshold']+=[threshold]
                params_history['Accuracy']+=[accuracy]
        else: 
            accuracy=kfold_score(X, y, clf, 5, metric='accuracy')
            params_history[keys[1]]+=[param.get(keys[1])]
            params_history[keys[0]]+=[param.get(keys[0])]
            params_history['Accuracy']+=[accuracy]
            
    return params_history

def plotting(x,y,z,xlabel,ylabel,title):
    print(z.shape)
    Z=z.reshape((len(y),len(x)))
    X, Y = np.meshgrid(x, y)
    ax = plt.axes(projection='3d')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel('Accuracy')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,cmap='viridis')
    ax.set_title(title)
    plt.savefig(title +'.png')


if __name__ == "__main__":
    print('------ import dataset ------')
    # filepath = '../dataset/data.csv'
    #filepath = os.path.join('C:' + os.sep + 'data' + os.sep + 'data.csv')
    #X, y = import_data(filepath)
    #X, y = make_moons(n_samples=1000, noise=0.1)

    filepath = "../dataset/X.csv"
    X = pd.read_csv(filepath, index_col=None)
    X = X.to_numpy()
    filepath = "../dataset/y.csv"
    y = pd.read_csv(filepath, index_col=None)
    y = np.ravel(y.to_numpy())
    #print(X.shape)
    #print(y.shape)
    # Randomize data
    p = np.random.permutation(len(y))
    X = X[p]
    y = y[p]
    #######logisticRegression
    print('-------Train LogisticRegression ------')
    lg_params={'penalty':['l1','l2', 'elasticnet'],'C':np.logspace(-3,-0.5,20)}
    lg_history=Tune_hyperparameter(LogisticRegression(solver='saga',l1_ratio=0.5),lg_params,X,y,5,thres=False)
    print('-------Plotting Hyperparameters vs Accuracy for LogisticRegression---')
    lg_x=range(len(lg_params['C']));lg_y=range(len(lg_params['penalty']));lg_z=np.array(lg_history['Accuracy'])
    lg_xlabel='C';lg_ylabel='Penalty';lg_title='Accuracy vs Hyperparameters for Logistic Regression'
    plotting(lg_x,lg_y,lg_z,lg_xlabel,lg_ylabel,lg_title) 
    #######SVM
    print('-------Train SVM-----')
    svm_params={'kernel':['linear', 'rbf'],'thres':np.linspace(0.2,0.7,30)}
    svm_history=Tune_hyperparameter(SVC(C=1.0,probability=True,gamma='auto'),svm_params,X,y,5,thres=True)
    print('-------Plotting Hyperparameters vs Accuracy for SVM--')
    svm_x=range(len(svm_params['kernel']));svm_y=range(len(svm_params['thres']));svm_z=np.array(svm_history['Accuracy'])
    svm_xlabel='Kernel';svm_ylabel='Threshold';svm_title='Accuracy vs Hyperparameters for SVM'
    plotting(svm_x,svm_y,svm_z,svm_xlabel,svm_ylabel,svm_title)
    
    #######Batchsize
    print('-------Batchsize vs Error-----')
    samplesize=range(3,15,2)
    clf=SVC(random_state=0,kernel='linear',C=1.0,gamma='auto')
    Tune_batchsize(samplesize,clf,X,y)
    
    ##### Neural Network
    print('------ train neural network ------')
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2)
    clf = MLPClassifier()
    cv_technique = 'k-fold'
    metric = 'accuracy'
    layers = [(10,), (10,10), (10,10,10), (50,), (50, 50), (50,50,50), 
    (100,), (100,100), (100,100,100), (200,), (200, 200), (200,200,200)]
    kernels = ['lbfgs']
    random_state = [42]
    parameter_grid = {'hidden_layer_sizes': layers, 'solver': kernels, 'random_state': random_state}
    print('------ training ------')
    clf, params, scores, best_param, best_score, nn_history = cross_validation_train(X_train_val, y_train_val, clf,
                                                                         parameter_grid, cv_technique=cv_technique,
                                                                         metric=metric)
    # model performance on the test set
    test_score = test_classifier(X_test, y_test, clf, metrics='accuracy')
    print('Best parameter: ', best_param)
    print('Best score: ', best_score)
    print('test score: ', test_score)
    print('-------Plotting Hyperparameters vs Accuracy for Neural Network--')
    nn_x=[1, 2, 3];nn_y=[10, 50, 100, 200];nn_z=np.array(nn_history['Accuracy'])
    nn_xlabel='Num of layers';nn_ylabel='Num of neurons';nn_title='Accuracy vs Hyperparameters for Neural Network'
    plotting(nn_x,nn_y,nn_z,nn_xlabel,nn_ylabel,nn_title)
