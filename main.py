from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC


def preprocess(filepath):
    # input:
    #     filepath: a path string of the dataset
    # output:
    #     X: a 2000*m Pandas Dataframe
    #     y: a 2000*1 Pandas Series
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
    # output:
    #     clf: a trained classifier
    return clf


def test_classifier(X_test, y_test, clf, metrics='accuracy'):
    # input:
    #     X_test, y_test: test (or validation) set
    #     clf: a trained classifier, SVM or NN or LR
    #     metrics: 'accuracy' or 'precision' or 'recall' or 'roc_auc' or ...
    # output:
    #     score: performance score on the test set
    return score


def cross_validation_score(X_train, y_train, clf, cv=5, metric='accuracy'):
    # input:
    #     X, y: training set
    #     clf: a sklearn classifier, SVM or NN or LR
    #     cv: number of folds
    #     metrics: 'accuracy' or 'precision' or 'recall' or 'roc_auc' or ...
    # output:
    #     average_score: mean of a list of scores get from test_classifier().
    return average_score


def cross_validation_train(X_train, y_train, clf, parameter_grid, cv=5, metric='accuracy'):
    return clf, params, scores, best_param, best_score


def bootstrap_socre(X, y, B=5, model='SVM', metric='accuracy'):
    return average_score


if __name__ == "__main__":
    # preprocess data
    filepath = 'dataset/books/'
    X, y = preprocess(filepath)
    # train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # initialize classifier
    clf = get_SVC()
    # cross validation train
    metric = 'accuracy'
    cv = 5
    clf, params, scores, best_param, best_score = cross_validation_train(X_train, y_train, clf, cv=cv, metric=metric)
    # model performance on the test set
    test_score = test_classifier(X_test, y_test, clf, metrics='accuracy')
