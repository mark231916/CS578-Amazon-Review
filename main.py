from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier
from itertools import product
import numpy as np
import math
import pandas as pd
from random import choices
import os
import random


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
    return accuracy_score(y_test, y_pred)


def kfold_score(X_train_val, y_train_val, clf, k=5, metric='accuracy'):
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
            score = kfold_score(X_train_val, y_train_val, clf, k=k, metric=metric)
        elif (cv_technique == 'bootstrap'):
            score = bootstrap_score(X_train_val, y_train_val, clf, B=B, metric=metric)
        print('cv score: {}'.format(score))
        params.append(param)
        scores.append(score)
        if (score > best_score):
            best_score = score
            best_param = param
    clf.set_params(**best_param)
    train_classifier(X_train_val, y_train_val, clf)
    return clf, params, scores, best_param, best_score



if __name__ == "__main__":
    print('------ import dataset ------')
    # filepath = '../dataset/data.csv'
    #filepath = os.path.join('C:' + os.sep + 'data' + os.sep + 'data.csv')
    #X, y = import_data(filepath)
    #X, y = make_moons(n_samples=1000, noise=0.1)

    filepath = "dataset/X.csv"
    X = pd.read_csv(filepath, index_col=None)
    X = X.to_numpy()
    filepath = "dataset/y.csv"
    y = pd.read_csv(filepath, index_col=None)
    y = np.ravel(y.to_numpy())
    #print(X.shape)
    #print(y.shape)
    # Randomize data
    p = np.random.permutation(len(y))
    X = X[p]
    y = y[p]
    print('------ split into training set and test set ------')
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2)
    #print(X_train_val)
    """ print('------ train classifier ------')
    clf = get_SVC()
    cv_technique = 'bootstrap'
    B = 5
    metric = 'accuracy'
    Cs = [0.001, 0.01, 0.1, 1, 10]
    kernels = ['linear', 'rbf']
    parameter_grid = {'C': Cs, 'kernel': kernels}
    print('------ training ------')
    clf, params, scores, best_param, best_score = cross_validation_train(X_train_val, y_train_val, clf,
                                                                         parameter_grid, cv_technique=cv_technique,
                                                                         metric=metric)
    # model performance on the test set
    test_score = test_classifier(X_test, y_test, clf, metrics='accuracy')
    print('Best parameter: ', best_param)
    print('Best score: ', best_score)
    print('test score: ', test_score) """

    print('------ train neural network ------')
    clf = MLPClassifier()
    cv_technique = 'k-fold'
    metric = 'accuracy'
    layers = [(10,), (10,10), (10,10,10), (50,), (50, 50), (50,50,50), 
    (100,), (100,100), (100,100,100), (200,), (200, 200), (200,200,200)]
    kernels = ['lbfgs']
    random_state = [42]
    parameter_grid = {'hidden_layer_sizes': layers, 'solver': kernels, 'random_state': random_state}
    print('------ training ------')
    clf, params, scores, best_param, best_score = cross_validation_train(X_train_val, y_train_val, clf,
                                                                         parameter_grid, cv_technique=cv_technique,
                                                                         metric=metric)
    # model performance on the test set
    test_score = test_classifier(X_test, y_test, clf, metrics='accuracy')
    print('Best parameter: ', best_param)
    print('Best score: ', best_score)
    print('test score: ', test_score)
