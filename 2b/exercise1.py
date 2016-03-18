from sklearn import metrics
import numpy as np
import pandas
from matplotlib import pyplot 

result = pandas.read_csv('classification_result.csv', sep = ',')

color = ['bo','ro','go','yo', 'mo']
classifiers = ['A', 'B']

thresholds = [0.2, 0.4, 0.5]
#thresholds = [0.5]
ground_truth = result['true']

pos = neg = 0.5
c_fn = 5
c_fp = 1
x = np.array([0.0,1.0])

convex_x = [0]
convex_y = [0]

tprs = []
fprs = []

for i in range(len(classifiers)):
    classifier_result = result[classifiers[i]].as_matrix()
    tpr = np.divide(1.0 * np.count_nonzero(classifier_result * ground_truth), len(classifier_result))
    fpr = np.divide(1.0 * np.sum(np.array(classifier_result > ground_truth)), len(classifier_result))
    cost = pos * (1 - tpr) * c_fn + neg * fpr * c_fp
    print '===classifier %s===' % classifiers[i]
    print 'true positive rate= %s' % tpr
    print 'false positive rate= %s' % fpr
    print 'cost = %s' % cost 
    pyplot.plot(fpr, tpr, color[i])
    pyplot.text(fpr+0.01, tpr+0.01, classifiers[i])
    tprs.append(tpr)
    fprs.append(fpr)

    #y = np.array(c_fp * x / c_fn + (tpr - fpr * c_fp / c_fn))
    #pyplot.plot(x, y)

for i in range(len(thresholds)):
    classifier_result = np.array(result['C'] >= thresholds[i]) * 1
    tpr = np.divide(1.0 * np.count_nonzero(classifier_result * ground_truth), len(classifier_result))
    fpr = np.divide(1.0 * np.sum(np.array(classifier_result > ground_truth)), len(classifier_result))
    cost = pos * (1 - tpr) * c_fn + neg * fpr * c_fp
    print '===classifier C, thresholds =  %s===' % thresholds[i]
    print 'true positive rate= %s' % tpr
    print 'false positive rate= %s' % fpr
    print 'cost = %s' % cost 
    pyplot.plot(fpr, tpr, color[2])
    pyplot.text(fpr+0.01, tpr+0.01, 'C' + str(thresholds[i]))

    if thresholds[i] == 0.5:
        convex_y.append(tpr)
        convex_x.append(fpr)

    #y = np.array(c_fp * x / c_fn + (tpr - fpr * c_fp / c_fn))
    #pyplot.plot(x, y)

convex_x.append(fprs[1])
convex_y.append(tprs[1])
convex_x.append(1)
convex_y.append(1)
pyplot.plot(convex_x, convex_y)

pyplot.title('ROC diagram of A,B and C with thresholds = 0.5')
pyplot.xlabel('false positive')
pyplot.ylabel('true positive')
pyplot.xlim(0,1)
pyplot.ylim(0,1)
pyplot.show()
