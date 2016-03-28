from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets import make_classification
from sklearn.svm import SVC
import sklearn.cross_validation as cv
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
from sklearn.metrics import roc_auc_score, roc_curve, auc


def create_data(num_samples,num_features):

    #return make_blobs(n_samples=num_samples,n_features=num_features)
    return make_classification(n_samples=num_samples,n_features=num_features,n_redundant=0,random_state=317)

def run():
    X,y = create_data(1000,2)
    print(X[1:10])
    print(y[1:10])
    print(y.shape)
    X_train,X_test,y_train,y_test = cv.train_test_split(X, y, test_size=.2)
    linear_svm = SVC(kernel="linear")
    rbf_svm = SVC(kernel="rbf")
    poly_svm = SVC(kernel="poly",degree=3)
    classifier_names= ['linear','polynomial','rbf']

    # step size for mesh
    h=.2
    #create mesh to plot in .point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    

    for i,clf in enumerate((linear_svm,poly_svm,rbf_svm)):
        scores = cv.cross_val_score(clf,X,y,cv =10)
        print("Mean Validation Score of with %s kernel: %0.2f (+/- %0.2f)" % (classifier_names[i],scores.mean(), scores.std() * 2))

        clf.fit(X,y)
        y_predicted = clf.predict(X)
        print("AUC under ROC curve with %s kernel : %.2f"%(classifier_names[i],roc_auc_score(y,y_predicted)))

        fpr,tpr,thresholds = roc_curve(y,y_predicted)
        roc_auc = auc(fpr,tpr)

        plt.subplot(2,2,4)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        plt.plot(fpr, tpr, lw=1, label='ROC with %s kernel (AUC = %0.2f)' % (classifier_names[i], roc_auc))


        plt.subplot(2, 2, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

        # Plot also the training points
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title("SVM with %s kernel (Mean Validation Score = %.2f)"%(classifier_names[i],scores.mean()))


    plt.subplot(2,2,4)
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")


    #plt2.show()
    plt.show()


if __name__ == "__main__":
    run()