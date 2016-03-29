import pandas as pd
from sklearn.svm import SVC
import numpy as np
from time import time
from operator import itemgetter
from matplotlib import pyplot as plt
from matplotlib import cm as CM
from matplotlib import mlab as ML
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import StratifiedShuffleSplit

def load_data_csv():
    result = pd.read_csv('ionosphere.csv', sep = ',')
    X = result.iloc[:,0:34]
    y = result['class']
    attribute_names = X.columns.values.tolist()
    
    return np.array(X), np.array(y), np.array(attribute_names)

def generate_grid(n_bins):
    r = C = np.logspace(-15, 15, n_bins, base = 2)
    exponents = np.linspace(-15, 15, n_bins)
    return C,r,exponents

def do_grid_search(X_train, y_train, X_test, y_test, r, exponents, param_grid):
    clf = SVC(probability = True)

    grid_search = GridSearchCV(clf, param_grid=param_grid, cv = 10)
    grid_search.fit(X_train, y_train)

    y_predict = grid_search.predict_proba(X_test)
    y_predict_score = np.array([y[1] for y in y_predict]) 

    print 'param = %s' % grid_search.best_params_
    print 'Grid Search score = %s' % grid_search.score(X_test, y_test)
    print 'AUC Score = %s' % roc_auc_score(y_test, y_predict_score)

def auc_by_grid_search(X_train, y_train, param_grid):
    clf = SVC(probability = True)

    grid_search = GridSearchCV(clf, param_grid=param_grid, cv = 10, scoring = 'roc_auc')
    grid_search.fit(X_train, y_train)
    
    scores = [score[1] for score in grid_search.grid_scores_]

    print 'param = %s' % grid_search.best_params_
    print 'Grid Search score = %s' % grid_search.best_score_

    return scores


def optimize_gamma(X_train, y_train, X_test, y_test, r, exponents):
    param_grid = {'gamma':r}
    do_grid_search(X_train, y_train, X_test, y_test, r, exponents, param_grid)

    auc_scores = auc_by_grid_search(X_train, y_train, param_grid)
    #for gamma_value in r:
    #    clf = SVC(gamma = gamma_value, probability = True)
    #    clf.fit(X_train, y_train)
    #    y_predict = clf.predict_proba(X_test)
    #    y_predict_score = np.array([y[1] for y in y_predict]) 
    #    auc_scores.append(roc_auc_score(y_test, y_predict_score))

    plt.plot(exponents, auc_scores, label = 'AUC scores of different gamma values')
    plt.xlabel('exponential values of gamma')
    plt.ylabel('auc scores')
    plt.axis([exponents.min(), exponents.max(), 0, 1])
    plt.show()

def optimize_both(X_train, y_train, X_test, y_test, r, exponents):
    start = time()
    param_grid = {'C': r,'gamma':r}
    do_grid_search(X_train, y_train, X_test, y_test, r, exponents, param_grid)
    print 'Running time = %f' % (time() - start)


def plot_C_gamma(X_train, y_train, r, exponents):
    param_grid = {'C': r,'gamma':r}
    auc_scores = auc_by_grid_search(X_train, y_train, param_grid)
    #for c in r:
    #    for gamma_value in r:
    #        clf = SVC(C = c, gamma = gamma_value, probability = True)
    #        clf.fit(X_train, y_train)
    #        y_predict = clf.predict_proba(X_test)
    #        y_predict_score = np.array([y[1] for y in y_predict]) 
    #        auc_scores.append(roc_auc_score(y_test, y_predict_score))
            
    scores = np.array(auc_scores).reshape(len(r), len(r))
    gridsize = len(r)^2
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot)
    plt.xticks(np.arange(len(r)), np.round(exponents), rotation=45)
    plt.yticks(np.arange(len(r)), np.round(exponents))
    plt.xlabel('Exponential values of gamma')
    plt.ylabel('Exponential values of C')

    cb = plt.colorbar()
    cb.set_label('mean value')
    plt.show()   

def run():
    C, r, exponents = generate_grid(20)
    X, y, attribute_names = load_data_csv()
    sss = StratifiedShuffleSplit(y, 1, test_size=0.2, random_state=0)

    for train_index, test_index in sss:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    #optimize_gamma(X_train, y_train, X_test, y_test, r, exponents)
    optimize_both(X_train, y_train, X_test, y_test, r, exponents)
    plot_C_gamma(X, y, r, exponents)
    

if __name__ == '__main__':
    run()
