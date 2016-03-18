from openml.apiconnector import APIConnector
import pandas as pd
import os
from sklearn import cross_validation as cv
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from time import time
from operator import itemgetter
from sklearn import tree
from scipy.stats import randint as sp_randint
import numpy as np

def load_data(dataset_id):
    #openml connection
    home_dir = os.path.expanduser("~")
    openml_dir = os.path.join(home_dir, "openml")
    cache_dir = os.path.join(openml_dir, "cache")
    with open(os.path.join(openml_dir, "apikey.txt"), 'r') as fh:
        key = fh.readline().rstrip('\n')
    openml = APIConnector(cache_directory=cache_dir, apikey=key)
    dataset = openml.download_dataset(dataset_id)
    # load data into panda dataframe
    X, y, attribute_names = dataset.get_dataset(target=dataset.default_target_attribute, return_attribute_names=True)

    print("no. of samples :"+str(len(X)))
    return (X,y,attribute_names)

def histogram(data,color):
    n, bins, patches = plt.hist(data, facecolor=color)
    plt.xlabel('class')
    plt.grid(True)
    plt.show()

def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (cross validation score: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

def runOptimization(optimizer, n_iter_search,n_top,optimizationAlgorithm,estimator,X,y):
    iteration_index = []
    mean_validation_scores = []
    start = time()
    optimizer.fit(X, y)
    if optimizationAlgorithm == "Grid Search":
        numParamSettings = len(optimizer.grid_scores_)
    else:
        numParamSettings = n_iter_search

    print(" ---------------------------------------------------------------------------")
    print("%s with %s took %.2f seconds for %d candidates"
      " parameter settings." % (optimizationAlgorithm,estimator,(time() - start), numParamSettings))
    report(optimizer.grid_scores_,n_top)

    for i, score in enumerate(optimizer.grid_scores_):
        iteration_index.append(i)
        mean_validation_scores.append(score.mean_validation_score)
    return (iteration_index,mean_validation_scores)

def run():
    X,y,attribute_names = load_data(32)

    for j in range(10):
        print(str(j)+"  "+str(sum( list(  1 if i==j else 0 for i in y))/ len(y)))

    X_train, X_test, y_train, y_test= cv.train_test_split(X,y,stratify = y,test_size=.2)

    for j in range(10):
        print(str(j)+"  "+str(sum( list(  1 if i==j else 0 for i in y_test))/ len(y_test)))

    sss =  cv.StratifiedShuffleSplit(y, 1, test_size=0.2, random_state=0)


    print("no. of stratified samples: "+str(len(sss)))
    for train,test in sss:
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
    for j in range(10):
        print(str(j)+"  "+str(sum( list(  1 if i==j else 0 for i in y_test))/ len(y_testd)))
    exit(0)

    # linear_svm
    clf = svm.LinearSVC()
    param_dist_linearSV =  {'C': np.logspace(-4.5, -2, 10),
                            "loss": [ "hinge", "squared_hinge" ],
                            "multi_class": ["ovr", "crammer_singer"]
                            }
    n_iter_search = 10
    random_search_lsvm = RandomizedSearchCV(clf, param_distributions=param_dist_linearSV,
                                   n_iter=n_iter_search)
    #runOptimization(random_search_lsvm, n_iter_search,n_iter_search,"Random Search","Linear Svm",X_train, y_train)


    # param_dist_tree = {"max_depth": sp_randint(6, 30),
    #                   "max_features": sp_randint(1, 10),
    #                   "min_samples_split": sp_randint(1, 10),
    #                   "min_samples_leaf": sp_randint(5, 20),
    #                   "criterion": ["gini", "entropy"]
    #                   }
    param_dist_tree = {"max_depth": [6,None],
                       "max_features": [1, 5,8, 10],
                       "min_samples_split": [5, 10, 15],
                       "min_samples_leaf": [5,10,15,20],
                       "criterion": ["gini", "entropy"]
                      }
    n_iter_search = 192
    clfTree1 = tree.DecisionTreeClassifier()
    random_search_tree = RandomizedSearchCV(clfTree1, param_distributions=param_dist_tree,n_iter=n_iter_search)
    randomIteration,randomScores = runOptimization(random_search_tree, n_iter_search,1,"Random Search","Decision Tree",X_train, y_train)



    param_grid_tree = {"max_depth": [6,None],
                       "max_features": [1, 5,8, 10],
                       "min_samples_split": [5, 10, 15],
                       "min_samples_leaf": [5,10,15,20],
                       "criterion": ["gini", "entropy"]
                      }
    clfTree2 = tree.DecisionTreeClassifier()
    grid_search_tree = GridSearchCV(clfTree2, param_grid=param_dist_tree)
    gridIteration,gridScores = runOptimization(grid_search_tree, n_iter_search,1,"Grid Search","Decision Tree",X_train, y_train)



    plt.plot(gridIteration,gridScores,"r-")
    plt.plot(randomIteration,randomScores,"b-")
    plt.legend(['GridSearch','Random Search'])
    plt.show()


if __name__ == "__main__":
    run()