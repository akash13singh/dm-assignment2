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


def visualize_tree(clf, feature_names, class_names):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open("dt.dot", 'w') as f:
        tree.export_graphviz(clf, out_file=f,feature_names=feature_names, class_names=class_names, filled=True, rounded=True,  special_characters=True)
    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
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


'''
# histogram of class variable
n, bins, patches = plt.hist(lymph['class'], facecolor='green')
plt.xlabel('class')
plt.grid(True)
plt.show()
'''
#remove classes
lymph = lymph[lymph['class'].isin([1,2])]
print(len(lymph))



plt.figure()
parallel_coordinates(lymph, 'class',colormap='gist_rainbow')
plt.xticks(rotation='vertical')
#plt.show()

'''
clf = tree.DecisionTreeClassifier()
X_train, X_test, y_train, y_test = train_test_split(X, y)
clf.fit(X_train, y_train)
print("Training error =", zero_one_loss(y_train, clf.predict(X_train)))
print("Test error =", zero_one_loss(y_test, clf.predict(X_test)))
visualize_tree(clf,attribute_names,['1','2'])
'''

clf = tree.DecisionTreeClassifier()
clf.fit(lymph.iloc[:,0:18],lymph.iloc[:,18:])
visualize_tree(clf,attribute_names,['1','2'])




