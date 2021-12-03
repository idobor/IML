#################################
# Your name: Ido Borenstein
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib

import numpy as np
from numpy.core.fromnumeric import shape, size
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing
import matplotlib.pyplot as plt
import warnings
np.seterr(invalid='ignore')
warnings.filterwarnings('ignore')



"""
Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""


def helper_hinge():
    mnist = fetch_openml('mnist_784',as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def helper_ce():
    mnist = fetch_openml('mnist_784',as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:8000] != 'a'))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[8000:10000] != 'a'))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = labels[train_idx[:6000]]

    validation_data_unscaled = data[train_idx[6000:8000], :].astype(float)
    validation_labels = labels[train_idx[6000:8000]]

    test_data_unscaled = data[8000 + test_idx, :].astype(float)
    test_labels = labels[8000 + test_idx]

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements Hinge loss using SGD.
    """
    w=np.zeros(len(data[0]))
    eta_t=eta_0
    n=len(data)
    for t in range(T):
      eta_t=eta_0/(t+1)
      i=numpy.random.randint(n)
      if np.dot(np.dot(labels[i],w),data[i])<1:
        w=np.dot((1-eta_t),w)+eta_t*C*np.dot(labels[i],data[i])
      else:
        w=np.dot((1-eta_t),w)
    return w
      

    

def SGD_ce(data, labels, eta_0, T):
    """
    Implements multi-class cross entropy loss using SGD.
    """
    W = np.zeros(shape=(10,len(data[0])))
    n=len(data)
    eta_t=eta_0
    for t in range(T):
        eta_t = eta_0/(t+1)
        i = numpy.random.randint(n)
        G = compute_gradient(W,data[i],labels[i])
        W = W - np.dot(eta_t,G)

    return W



#################################

# Place for additional code

"""
This function returns the gradient as a np vector
"""
def compute_gradient(W_s,x,y):
    L = len(W_s)
    G = np.ndarray(shape=W_s.shape)
    for j in range(L):
        w_k = W_s[j]
        if j==int(y):
            G[j] = (np.dot(np.exp(np.dot(w_k,x)),x))/(np.sum([np.exp(np.dot(W_s[i],x)) for i in range(L)])) - x
            
            
        else:
            #Without LSE:
            G[j] = (np.dot(np.exp(np.dot(w_k,x)),x))/(np.sum([np.exp(np.dot(W_s[i],x)) for i in range(L)]))
    return G



def task_1a():
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper_hinge()
    etas = [10**(i-5) for i in range(11)]
    accuracy_list = []
    for i,eta in enumerate(etas):
        accuracy=0
        for j in range(10):
            w = SGD_hinge(train_data,train_labels,1,eta,1000)
            accuracy += get_accuracy_hinge(w,validation_data,validation_labels)
        accuracy_list.append(accuracy/10)
    best_eta=etas[np.argmax(accuracy_list)]
    plt.xscale('log')
    plt.xlabel("eta")
    plt.ylabel("accuracy")
    plt.plot(etas,accuracy_list)
    plt.show()


def task_1b():
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper_hinge()
    C_s = [10**(i-5) for i in range(11)]
    accuracy_list = []
    for i, c in enumerate(C_s):
        accuracy = 0 
        for j in range(10):
            W = SGD_hinge(train_data,train_labels,c,1,1000)
            accuracy += get_accuracy_hinge(W,validation_data,validation_labels)
        accuracy_list.append(accuracy/10)
    best_C=C_s[np.argmax(accuracy_list)]
    plt.xscale('log')
    plt.xlabel("C")
    plt.ylabel("accuracy")
    plt.plot(C_s,accuracy_list)
    plt.show()
    return best_C


def task_1c():
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper_hinge()
    W = SGD_hinge(train_data,train_labels,0.0001,1,20000)
    accuracy = get_accuracy_hinge(W,test_data,test_labels)
    plt.imshow(W.reshape((28,28)), interpolation='nearest')
    plt.show()
    return accuracy

def task_2a():
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper_ce()
    etas = [10**(i-5) for i in range(11)]
    #etas = np.linspace(0.0000001,0.00001,10)
    accuracy_list = []
    for i,eta in enumerate(etas):
        accuracy=0
        for j in range(10):
            W = SGD_ce(train_data,train_labels,eta,1000)
            accuracy += get_accuracy_CE(W,validation_data,validation_labels)
        accuracy_list.append(accuracy/10)
    best_eta=etas[np.argmax(accuracy_list)]
    plt.xscale('log')
    plt.xlabel("eta")
    plt.ylabel("accuracy")
    plt.plot(etas,accuracy_list)
    plt.show()
    return best_eta


def task_2b():
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper_ce()
    W = SGD_ce(train_data,train_labels,0.00001,20000)
    accuracy = get_accuracy_CE(W,test_data,test_labels)
    
    fig = plt.figure(figsize=(20,10))
    fig.tight_layout(pad=5.0)
    for i in range(10):
        fig.add_subplot(5,2,i+1)
        plt.imshow(W[i].reshape((28,28)), interpolation='nearest')
        plt.title(str(i))
    plt.show()
    return accuracy


def my_sign(x):
    if x>0:
        return 1
    return -1
def is_correct_hinge(w,x,y):
    return my_sign(np.dot(w,x))==y
def get_accuracy_hinge(w,data,labels):
    accuracy=0
    for i,x in enumerate(data):
        if is_correct_hinge(w,x,labels[i]):
            accuracy+=1
    return accuracy/len(data)

def is_correct_CE(W,x,y):
    preditcion = get_prediction(W, x)
    return preditcion == int(y)


def get_prediction(W, x):
    return np.argmax([np.dot(W[i],x) for i in range(len(W))])


def get_accuracy_CE(W,data,labels):
    accuracy = 0
    for i,x in enumerate(data):
        if is_correct_CE(W,x,labels[i]):
            accuracy+=1
    return accuracy/len(data)

def get_accuracy_for_label(W,data,labels):
    results = {i:[0,0,0] for i in range(10)}
    for i,x in enumerate(data):
        results[int(labels[i])][1]+=1
        if is_correct_CE(W,x,labels[i]):
            results[int(labels[i])][0]+=1
    for res in results:
        results[res][2]=results[res][0]/results[res][1]
        results[res][0] = "Correct predictions: " + str(results[res][0])
        results[res][1] = "Set size: " + str(results[res][1])
        results[res][2] = "Score: " + str(results[res][2])

    return results
            




task_1a()
task_1b()
task_1c()
task_2a()
task_2b()
#################################