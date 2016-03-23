#from openml.apiconnector import APIConnector
import pandas as pd
import os
from sklearn import cross_validation as cv
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.metrics import classification_report,zero_one_loss
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

def load_data_csv():
    result = pd.read_csv('pendigits.csv', sep = ',')
    X = result.iloc[:,0:16]
    y = result['class']
    attribute_names = X.columns.values.tolist()

    return X, y, attribute_names

def histogram(data,color):
    n, bins, patches = plt.hist(data, facecolor=color)
    plt.xlabel('class')
    plt.grid(True)
    plt.show()

def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("cross-validation scores "+str(score.cv_validation_scores))
        print("Parameters: {0}".format(score.parameters))
        print("")

def runOptimization(optimizer, n_iter_search,n_top,optimizationAlgorithm,estimator,X,y):
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
    return optimizer

def getOptimizerScores(optimizer):
    iteration_index = []
    mean_validation_scores = []
    for i, score in enumerate(optimizer.grid_scores_):
        iteration_index.append(i)
        mean_validation_scores.append(score.mean_validation_score)
    return iteration_index,mean_validation_scores


def run():
    #X,y,attribute_names = load_data(32)
    X,y,attribute_names = load_data_csv()
    print(attribute_names)

    #print("% of different classes in dataset")
    #for j in range(10):
    #    print(str(j)+"  "+str(sum( list(  1 if i==j else 0 for i in y))/ len(y)))


    X_train, X_test, y_train, y_test= cv.train_test_split(X,y,stratify = y,test_size=.2,random_state=0)


    '''
    sss =  cv.StratifiedShuffleSplit(y, 1, test_size=0.2, random_state=0)
    print("no. of stratified samples: "+str(len(sss)))
    for train,test in sss:
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
    for j in range(10):
        print(str(j)+"  "+str(sum( list(  1 if i==j else 0 for i in y_test))/ len(y_test)))
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
    '''



    param_dist_tree = {"max_depth": sp_randint(5, 30),
                       "min_samples_split": sp_randint(5, 30),
                       "min_samples_leaf": sp_randint(5, 30),
                       "criterion": ["gini", "entropy"]
                      }

    n_iter_search = 20
    # random search with nested resampling/cv

    clf_tree1 = tree.DecisionTreeClassifier()
    random_search_tree1 = RandomizedSearchCV(clf_tree1, param_distributions=param_dist_tree,n_iter=n_iter_search, cv = 10)
    random_search_optimizer1 = runOptimization(random_search_tree1, n_iter_search,1,"Random Search with CV","Decision Tree",X_train, y_train)
    random_iteration1 , random_scores1 = getOptimizerScores(random_search_optimizer1)
    random_best1 = random_search_optimizer1.best_estimator_



    # random search with no rested resampling/cv
    #The mean validation score reported here is the error on test data
    sss =  cv.StratifiedShuffleSplit(y, 1, test_size=0.2, random_state=0)
    clf_tree2 = tree.DecisionTreeClassifier()
    random_search_tree2 = RandomizedSearchCV(clf_tree2, param_distributions=param_dist_tree,n_iter=n_iter_search, cv = sss)
    random_search_optimizer2 = runOptimization(random_search_tree2, n_iter_search,1,"Random Search without CV","Decision Tree",X, y)
    random_iteration2 , random_scores2 = getOptimizerScores(random_search_optimizer2)
    random_best2 = random_search_optimizer2.best_estimator_


    param_grid_tree = {"max_depth": [5,10,15],
                       "min_samples_split": [5,20,15],
                       "min_samples_leaf": [5,20,15,],
                       "criterion": ["gini", "entropy"]
                      }
    clf_tree3 = tree.DecisionTreeClassifier()
    grid_search_tree = GridSearchCV(clf_tree3, param_grid=param_grid_tree,cv = 10)
    grid_search_optimizer = runOptimization(grid_search_tree, n_iter_search,1,"Grid Search","Decision Tree",X_train, y_train)
    grid_iteration , grid_scores = getOptimizerScores(grid_search_optimizer)
    grid_best = grid_search_optimizer.best_estimator_


    plt.plot(grid_iteration,grid_scores,"r-",label="GridSearchScore")
    plt.plot(random_iteration1,random_scores1,"b-",label="RandomSearchScoreWithCV")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
    plt.xticks(np.arange(0,len(grid_scores),2))
    plt.yticks(np.arange(.5,1,.02))
    plt.show()

    '''
    plt.plot(random_iteration2,random_scores2,"b-",label="RandomSearchScoresWithoutCV")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
    plt.xticks(np.arange(0,len(random_scores2),2))
    plt.yticks(np.arange(.5,1,.02))
    plt.show()
    '''
    print("------Test Evaluation for Random Search With CV-------")
    predicted_random1 = random_best1.predict(X_test)
    print("Test Error: "+str(zero_one_loss(y_test,predicted_random1)))


    print("------Test Evaluation for Grid Search-------")
    predicted_grid = grid_best.predict(X_test)
    print("Test Error: "+str(zero_one_loss(y_test,predicted_grid)))

if __name__ == "__main__":
    run()
