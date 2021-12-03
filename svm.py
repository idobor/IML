#################################
# Your name: Ido Borenstein
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score

"""
Please use the provided functions signature for the SVM implementation.
Feel free to add functions and other code, and submit this file with the name svm.py
"""

# generate points in 2D
# return training_data, training_labels, validation_data, validation_labels
def get_points():
    X, y = make_blobs(n_samples=120, centers=2, random_state=0, cluster_std=0.88)
    return X[:80], y[:80], X[80:], y[80:]


def create_plot(X, y, clf):
    plt.clf()

    # plot the data points
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.PiYG)

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0] - 2, xlim[1] + 2, 30)
    yy = np.linspace(ylim[0] - 2, ylim[1] + 2, 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])


def train_three_kernels(X_train, y_train, X_val, y_val):
    """
    Returns: np.ndarray of shape (3,2) :
                A two dimensional array of size 3 that contains the number of support vectors for each class(2) in the three kernels.
    """
    result = np.ndarray(shape=(3,2))
    #Linear:
    linear = svm.SVC(C=1000,kernel='linear')
    linear.fit(X_train,y_train)
    result[0] = linear.n_support_
    create_plot(X_train,y_train,linear)
    plt.show()
    quar = svm.SVC(C=1000,kernel='poly',degree=2)
    quar.fit(X_train,y_train)
    result[1] = quar.n_support_
    create_plot(X_train,y_train,quar)
    plt.show()
    rbf = svm.SVC(C=1000,kernel='rbf')
    rbf.fit(X_train,y_train)
    result[2] = rbf.n_support_
    create_plot(X_train,y_train,rbf)
    plt.show()
    print(result)
    return result




def linear_accuracy_per_C(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    C_s = [10**i for i in range(-5,6)]
    accuracy_res_validation = []
    accuracy_res_train = []
    figure(figsize=(5, 3), dpi=80)

    for i,c in enumerate(C_s):
      linear = svm.SVC(C=c,kernel="linear")
      linear.fit(X_train,y_train)
      preds_valdiation = linear.predict(X_val)
      preds_train = linear.predict(X_train)
      accuracy_res_validation.append(accuracy_score(preds_valdiation,y_val))
      accuracy_res_train.append(accuracy_score(preds_train,y_train))
      create_plot(X_train,y_train,linear)
      plt.title(str(c))
      plt.show()
    
    plt.plot(C_s,accuracy_res_validation,label="validation")
    plt.plot(C_s,accuracy_res_train,label="train")
    plt.xscale("log")
    plt.xlabel("C")
    plt.xlabel("Accuracy")
    leg = plt.legend(loc='center right')

    plt.show()
    best_c_validation = C_s[np.argmax(accuracy_res_validation)]
    # linear = svm.SVC(C=best_c_validation,kernel=linear)
    # linear.fit(X_train,y_train)

    print(best_c_validation)

    return np.array(accuracy_res_validation)
      



def rbf_accuracy_per_gamma(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    gammas = [10**i for i in np.arange(-1,1.2,0.2)]
    #gammas =[0.000000001*i for i in range(1,1000)]
    gammas = [10**i for i in range(-5,6)]
    accuracy_res_validation = []
    accuracy_res_train = []
    for g in gammas:
      rbf = svm.SVC(C=10,kernel="rbf",gamma=g)
      rbf.fit(X_train,y_train)
      preds_validation = rbf.predict(X_val)
      preds_train = rbf.predict(X_train)
      accuracy_res_validation.append(accuracy_score(preds_validation,y_val))
      accuracy_res_train.append(accuracy_score(preds_train,y_train))
      create_plot(X_train,y_train,rbf)
      plt.title(str(g))
      plt.show()
    plt.plot(gammas,accuracy_res_validation,label="Validation")
    plt.plot(gammas,accuracy_res_train,label="Train")
    plt.xscale("log")
    plt.xlabel("gamma")
    plt.ylabel("Accuracy")
    #plt.ylim([0.97,0.98])
    leg = plt.legend(loc='center right')
    plt.show()
    print(gammas[np.argmax(accuracy_res_validation)])
    print
    return np.array(accuracy_res_validation)

X_train, y_train, X_val, y_val = get_points()
#train_three_kernels(X_train, y_train, X_val, y_val)
#linear_accuracy_per_C(X_train, y_train, X_val, y_val)
rbf_accuracy_per_gamma(X_train, y_train, X_val, y_val)