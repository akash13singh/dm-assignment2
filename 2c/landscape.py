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

def optimize_gamma(X_train, y_train, X_test, y_test, r):
    auc_scores = []
    for gamma_value in r:
        clf = SVC(gamma = gamma_value, probability = True)
        clf.fit(X_train, y_train)
        y_predict = clf.predict_proba(X_test)
        y_predict_score = np.array([y[1] for y in y_predict]) 
        auc_scores.append(roc_auc_score(y_test, y_predict_score))

    plt.plot(r, auc_scores, label = 'AUC scores of different gamma values')
    plt.xlabel('gamma values')
    plt.ylabel('auc scores')
    plt.show()

def optimize_both(X_train, y_train, X_test, y_test, r):
    start = time()
    param_grid = {'C': r,'gamma':r}
    clf = SVC(probability = True)

    grid_search = GridSearchCV(clf, param_grid=param_grid, cv = 10)
    grid_search.fit(X_train, y_train)

    y_predict = grid_search.predict_proba(X_test)
    y_predict_score = np.array([y[1] for y in y_predict]) 

    print 'Running time = %f' % (time() - start)
    print 'Grid Search score = %s' % grid_search.score(X_test, y_test)
    print 'AUC Score = %s' % roc_auc_score(y_test, y_predict_score)


def plot_C_gamma(X_train, y_train, X_test, y_test, r):
    auc_scores = []
    for c in r:
        for gamma_value in r:
            clf = SVC(gamma = gamma_value, probability = True)
            clf.fit(X_train, y_train)
            y_predict = clf.predict_proba(X_test)
            y_predict_score = np.array([y[1] for y in y_predict]) 
            auc_scores.append(roc_auc_score(y_test, y_predict_score))
            
    gridsize = len(r)^2
    plt.hexbin(r, r, C=auc_scores, gridsize=gridsize, cmap=CM.jet, bins=None)
    plt.axis([r.min(), r.max(), r.min(), r.max()])

    cb = plt.colorbar()
    cb.set_label('mean value')
    plt.show()   

def run():
    r = np.logspace(-15, 15, 5, base = 2)
    X, y, attribute_names = load_data_csv()
    sss = StratifiedShuffleSplit(y, 1, test_size=0.2, random_state=0)

    for train_index, test_index in sss:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    #optimize_gamma(X_train, y_train, X_test, y_test, r)
    #optimize_both(X_train, y_train, X_test, y_test, r)
    plot_C_gamma(X_train, y_train, X_test, y_test, r)
    

if __name__ == '__main__':
    run()
