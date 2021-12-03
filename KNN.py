from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy.random
import numpy as np
from numpy.linalg import norm
from numpy import argsort
from collections import Counter
mnist = fetch_openml('mnist_784', as_frame=False)
data = mnist['data']
labels = mnist['target']
idx=numpy.random.RandomState(0).choice(70000,11000)
train=data[idx[:10000],:].astype(int)
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :].astype(int)
test_labels = labels[idx[10000:]]


#task a
def knn(k,train_data,train_labels, test_point):
  distance_metrics=norm(train_data-test_point,axis=1)
  indices_of_points=np.argsort(distance_metrics)[:k]
  outputs=Counter([train_labels[i] for i in indices_of_points])
  prediction=outputs.most_common(1)[0][0]
  return prediction


#task b
def check_accuracy(n,k):
  train_set=train[:n]
  correct_predictions=0
  for i,image in enumerate(test):
    if (knn(k,train_set,train_labels,image))==test_labels[i]:
      correct_predictions+=1
  return correct_predictions/len(test)

#task c
def plot_and_check_accuracy(k_range,n):
  results=[check_accuracy(n,i) for i in range(1,k_range+1)]
  print("The best k is:",end="")
  print(results.index(max(results))+1)
  plt.xlabel("k")
  plt.ylabel("Accuracy")
  plt.plot([k for k in range(1,k_range+1)],results)
  plt.show()

#task d
def plot_and_check_fixed_k(k):
  results=[check_accuracy(n,k) for n in range(100,5001,100)]
  plt.xlabel("n")
  plt.ylabel("Accuracy")
  plt.plot([n for n in range(100,5001,100)],results)
  plt.show()

def run_tasks():
  print(check_accuracy(1000,10))
  plot_and_check_accuracy(100,1000)
  plot_and_check_fixed_k(1)

run_tasks()








