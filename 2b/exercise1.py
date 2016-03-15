from sklearn import metrics
import numpy as np
import pandas
from matplotlib import pyplot 

result = pandas.read_csv('classification_result.csv', sep = ',')
color = ['b.','r.','g.','y.', 'm.']
classifiers = ['A', 'B', 'C']

true_labels = result["true"].as_matrix()
for i in range(len(classifiers)):
    ranks = result[classifiers[i]].as_matrix()
    
    fpr, tpr, thresholds = metrics.roc_curve(true_labels, ranks)
    print '=======Classifier %s ========' % classifiers[i]
    print 'false positive rate:  %s' % fpr
    print 'true positive rate: %s' % tpr
    print 'thresholds: %s' % thresholds
    
    pyplot.plot(fpr, tpr)

pyplot.show()
