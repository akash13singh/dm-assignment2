from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets import make_classification
from sklearn.svm import SVC
import sklearn.cross_validation as cv
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.grid_search import GridSearchCV

def create_data(num_samples,num_features):

    #return make_blobs(n_samples=num_samples,n_features=num_features)
    return make_classification(n_samples=num_samples,n_features=num_features,n_redundant=0,random_state=317)

def run():
    X,y = create_data(1000,2)


    gamma_range = np.logspace(start=-15,stop= 15,num =4, base=2)
    gamma_range = np.append(gamma_range,[2**30,2**(-30)])
    gamma_grid = dict(gamma=gamma_range)
    print(gamma_grid)

    C_range = np.logspace(start=-15,stop= 15,num =4, base=2)
    C_range = np.append(C_range,[2**20,2**-20])
    C_grid = dict(C=C_range)
    print(C_grid)

    plt.figure(figsize=(10, 7))
    h=.2
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

    grid = GridSearchCV(SVC(kernel="rbf"), param_grid=gamma_grid, cv=10)
    grid.fit(X,y)
    grid_scores = grid.grid_scores_
    print("The best parameters are %s with a score of %0.2f"% (grid.best_params_, grid.best_score_))
    #print(grid_scores)

    for i, score in enumerate(grid_scores):

        gamma = score.parameters['gamma']
        clf = SVC(kernel="rbf",gamma=gamma,probability=True)
        clf.fit(X,y)
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        y_predicted = clf.predict_proba(X)
        #print(score.cv_validation_scores)
        #print("(%d) γ=2^%d, C=%s, CV-Score = %.3f, accuracy=%.2f, AUC = %.3f" %(i+1,np.log2(gamma), "Default",score.mean_validation_score,accuracy_score(y,y_predicted),roc_auc_score(y,y_predicted)))

        # visualize decision function for these parameters
        plt.subplot(3, 4, i+1)
        plt.title("(%d) γ=2^%d C=%s CV-Score = %.3f AUC = %.3f"  % (i+1,np.log2(gamma), "Default",score.mean_validation_score,roc_auc_score(y,y_predicted[:,1])), size='medium')

        # visualize parameter's effect on decision function
        plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu_r)
        plt.xticks(())
        plt.yticks(())
        plt.axis('tight')


    grid2 = GridSearchCV(SVC(kernel="rbf"), param_grid=C_grid, cv=10)
    grid2.fit(X,y)
    grid_scores2 = grid2.grid_scores_
    print("The best parameters are %s with a score of %0.2f"% (grid2.best_params_, grid2.best_score_))

    for i, score in enumerate(grid_scores2):
        C = score.parameters['C']
        clf = SVC(kernel="rbf",C=C,probability=True)
        clf.fit(X,y)
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        y_predicted = clf.predict_proba(X)
        #print("(%d) C=2^%d, γ=%s, CV-Score = %.3f, accuracy=%.2f, AUC = %.3f" % ((i+1)+6,np.log2(C), "Default",score.mean_validation_score,accuracy_score(y,y_predicted),roc_auc_score(y,y_predicted)))

        # visualize decision function for these parameters
        plt.subplot(3, 4, (i+1)+6)
        plt.title("(%d) C=2^%d γ=%s CV-Score = %.3f AUC = %.3f" % ((i+1)+6,np.log2(C), "Default",score.mean_validation_score,roc_auc_score(y,y_predicted[:,1])), size='medium')

        # visualize parameter's effect on decision function
        plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu_r)
        plt.xticks(())
        plt.yticks(())
        plt.axis('tight')

    plt.show()

if __name__ == "__main__":
    run()
