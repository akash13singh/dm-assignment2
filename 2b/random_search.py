from openml.apiconnector import APIConnector
import pandas as pd
import os
from sklearn import cross_validation as cv
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.metrics import classification_report,zero_one_loss,accuracy_score
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from time import time
from operator import itemgetter
from sklearn import tree
from scipy.stats import randint as sp_randint
import numpy as np
import random

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

def run():
     X,y,attribute_names = load_data(32)
     randomSearch(X,y)

'''
param_dist_tree = {"max_depth": sp_randint(5, 30),
                       "min_samples_leaf": sp_randint(5,20),
                       "min_samples_split": sp_randint(5, 40),
                       # "criterion": ["gini", "entropy"]
                      }
'''

#function for parameter optimization on training data. Uses random search.
def randomSearch(X,y):
    random.seed(5)
    best_score = 0
    scores = []
    n_iter = 20
    X_train, X_test, y_train, y_test= cv.train_test_split(X,y,stratify = y,test_size=.2,random_state=0)
    for i in range(n_iter):
        depth = random.randint(5,30)
        samples_leaf = random.randint(5,30)
        samples_split = random.randint(5,30)
        criterion_value = random.randint(0,1)
        if criterion_value ==0:
            gini_entropy = "gini"
        else :
            gini_entropy = "entropy"
        clf_tree = tree.DecisionTreeClassifier(max_depth=depth,min_samples_leaf=samples_leaf , min_samples_split=samples_split, criterion=gini_entropy)
        clf_tree.fit(X_train,y_train)
        predicted_train = clf_tree.predict(X_train)
        score = accuracy_score(y_train,predicted_train)
        scores.append(score)
        print("score : %.4f , max_depth = %d, max_samples_split = %d, max_samples_leaf = %d and criterion =%s "%(score,depth,samples_split,samples_leaf,gini_entropy))
        if (score > best_score):
            best_depth = depth
            best_samples_leaf = samples_leaf
            best_samples_split = samples_split
            best_criterion = gini_entropy
            best_score = score
            best_model = clf_tree


    print("best model has score %.4f and max_depth = %d, max_samples_split = %d, max_samples_leaf = %d and criterion =%s"%(best_score,best_depth,best_samples_split,best_samples_leaf,best_criterion))
    predicted_test = best_model.predict(X_test)
    predicted_train = best_model.predict(X_train)
    print("The best model has a training set accuracy of %.4f "%(accuracy_score(y_train,predicted_train)))
    print("The best model has a test set accuracy of %.4f "%(accuracy_score(y_test,predicted_test)))
    print(len(scores))
    plt.plot(np.arange(1,n_iter+1,1),scores,"b-",label="RandomSearchScoresWithoutCV")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
    plt.xticks(np.arange(1,n_iter+1,2))
    plt.yticks(np.arange(.5,1,.02))
    plt.show()

if __name__ == "__main__":
    run()