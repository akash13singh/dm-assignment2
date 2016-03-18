from sklearn.datasets.samples_generator import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
from matplotlib import pyplot 
from sklearn.utils import resample
from sklearn import tree
import numpy as np
from numpy import arange
import sys

def bootstrapping_score(classifier, dataset, true_labels, repeats):
    scores = []
    for k in range(repeats):
        #indices = np.random.choice(arange(len(dataset)), len(dataset) - 1)
        indices = resample(arange(len(dataset)))
        training_set = np.array([dataset[i] for i in indices])
        training_labels = np.array([true_labels[i] for i in indices])

        test_set = [dataset[i] for i in range(len(dataset)) if i not in indices]
        test_labels = [true_labels[i] for i in range(len(true_labels)) if i not in indices]
        classifier.fit(training_set, training_labels)
        scores.append(classifier.score(test_set, test_labels))
    return np.array(scores)

def validation(classifier, dataset, true_labels, is_bootstrapping):
    if is_bootstrapping:
        return bootstrapping_score(classifier, dataset, true_labels, 100)
    else:
        return cross_validation.cross_val_score(classifier, dataset, true_labels, cv=10)

def knn_evaluation():
    print 'KNeighborsClassifier'
    np.random.seed(123)
    dataset,true_labels = make_blobs(n_samples=10000, n_features=2)
    color = ['r-', 'b-']
    methods = [True, False]

    for b in methods:
        print 'bootstrapping = %s' % methods[b]
        misclassification_rates = []
        min_rate = np.inf
        min_k = 0

        for i in range(1,51):
            neigh = KNeighborsClassifier(n_neighbors=i)
            scores = validation(neigh, dataset, true_labels, methods[b])
            misclassifications = 1 - scores
            misclassification_rates.append(np.average(misclassifications))
        
            if min_rate > misclassification_rates[i-1]:
                min_rate = misclassification_rates[i-1]
                min_k = i

        print 'minimum rate = %s' % min_rate
        print 'best k = %s' % min_k

        label = 'bootstrap' if methods[b] else 'cross-validation'
        pyplot.plot(range(1,51), misclassification_rates, color[b], label = label)
    
    pyplot.title('Mis-classification rates of KNeighborsClassifier')
    pyplot.xlabel('Values of k')
    pyplot.ylabel('Mis classification rates')
    pyplot.legend(loc = 'upper right')
    pyplot.show()

def tree_evaluation():
    print 'DecisionTreeClassifier'
    np.random.seed(123)
    dataset,true_labels = make_blobs(n_samples=10000, n_features=2)
    color = ['r-', 'b-']
    methods = [True, False]

    for b in methods:
        print 'bootstrapping = %s' % methods[b]
        misclassification_rates = []
        min_rate = np.inf
        min_k = 0

        for i in range(2,16):
            tree_classifier = tree.DecisionTreeClassifier(max_depth=i)
            scores = validation(tree_classifier, dataset, true_labels, methods[b])
            misclassifications = 1 - scores
            misclassification_rates.append(np.average(misclassifications))
        
            if min_rate > misclassification_rates[i-2]:
                min_rate = misclassification_rates[i-2]
                min_k = i

        print 'minimum rate = %s' % min_rate
        print 'best depth = %s' % min_k

        label = 'bootstrap' if methods[b] else 'cross-validation'
        pyplot.plot(range(2,16), misclassification_rates, color[b], label = label)
    
    pyplot.title('Mis-classification rates of DecisionTreeClassifier')
    pyplot.xlabel('Values of k')
    pyplot.ylabel('Mis classification rates')
    pyplot.legend(loc = 'upper left')
    pyplot.show()

if __name__ == '__main__':
    if sys.argv[1] == 'tree':
        tree_evaluation()
    else:
        knn_evaluation()
