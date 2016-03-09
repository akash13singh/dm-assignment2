from openml.apiconnector import APIConnector
import pandas as pd
import os
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from pandas.tools.plotting import parallel_coordinates
from sklearn import tree
from sklearn.cross_validation import train_test_split
import subprocess
from sklearn.metrics import zero_one_loss
from sklearn.metrics import classification_report
import numpy as np



def build_tree(clf,type,i,X_train, X_test, y_train, y_test,attribute_names,class_names):
    print("------------Run "+type+ "_"+str(i)+"----------")
    clf.fit(X_train, y_train)
    print("Training error =", zero_one_loss(y_train, clf.predict(X_train)))
    predicted_test = clf.predict(X_test)
    print("Test error =",zero_one_loss(y_test, predicted_test ) )
    figure_name = type+"_"+str(i)
    visualize_tree(clf,attribute_names,class_names,figure_name)
    print(classification_report(  y_test,predicted_test ))
    return zero_one_loss(y_test, predicted_test )

def visualize_tree(clf, feature_names, class_names,figure_name):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open("dt.dot", 'w') as f:
        tree.export_graphviz(clf, out_file=f,feature_names=feature_names, class_names=class_names, filled=True, rounded=True,  special_characters=True)
    command = ["dot", "-Tpng", "dt.dot", "-o", figure_name+".png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")


#openml connection
home_dir = os.path.expanduser("~")
openml_dir = os.path.join(home_dir, "openml")
cache_dir = os.path.join(openml_dir, "cache")
with open(os.path.join(openml_dir, "apikey.txt"), 'r') as fh:
    key = fh.readline().rstrip('\n')
openml = APIConnector(cache_directory=cache_dir, apikey=key)
dataset = openml.download_dataset(10)
dataset = openml.download_dataset(10)


# load data into panda dataframe
X, y, attribute_names = dataset.get_dataset(target=dataset.default_target_attribute, return_attribute_names=True)
lymph = pd.DataFrame(X, columns=attribute_names)
lymph['class'] = y
print(len(lymph))



# histogram of class variable
n, bins, patches = plt.hist(lymph['class'], facecolor='green')
plt.xlabel('class')
plt.grid(True)
#plt.show()

#remove classes
lymph = lymph[lymph['class'].isin([1,2])]
print(len(lymph))


plt.figure()
parallel_coordinates(lymph, 'class',colormap='gist_rainbow')
plt.xticks(rotation='vertical')
#plt.show()


seeds = [31,67,321,5,76,43,12,11]
class_names = ["metastases","malign lymph"]


clf = tree.DecisionTreeClassifier()
clf_extra = tree.ExtraTreeClassifier()
clf_pruned_cart = tree.DecisionTreeClassifier(min_samples_leaf=5, max_depth=5)
test_errors_CART = []
test_errors_Extra = []
test_errors_cart_pruned = []

for i in range(8):
    X_train, X_test, y_train, y_test = train_test_split(lymph.iloc[:,0:18],lymph.iloc[:,18:],test_size=.2,random_state=seeds[i])
    test_errors_CART.append(build_tree(clf,'cart',i,X_train, X_test, y_train, y_test,attribute_names,class_names))
    test_errors_Extra.append(build_tree(clf_extra,'extra',i,X_train, X_test, y_train, y_test,attribute_names,class_names))
    test_errors_cart_pruned.append(build_tree(clf_pruned_cart,'pruned_cart',i,X_train, X_test, y_train, y_test,attribute_names,class_names))

print("-----------------------------------------------------------")

print("Cart Average error :"+str(np.average(test_errors_CART)))
print("Cart Variance of errors :"+str(np.var(test_errors_CART)))

print("Extra Average error :"+str(np.average(test_errors_Extra)))
print("Extra Variance of errors :"+str(np.var(test_errors_Extra)))

print("Cart Pruned Average error :"+str(np.average(test_errors_cart_pruned)))
print("Cart Pruned Variance of errors :"+str(np.var(test_errors_cart_pruned)))