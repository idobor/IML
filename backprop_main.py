import backprop_data

import backprop_network
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import numpy as np
# np.seterr(invalid='ignore')
# warnings.filterwarnings('ignore')


training_data, test_data = backprop_data.load(train_size=10000,test_size=5000)


net = backprop_network.Network([784, 40, 10])

#net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1, test_data=test_data)

def task_a(training_data, test_data):
    rates = [10**i for i in range(-3,3)]
    train = [i for i in range(len(rates))]
    test = [i for i in range(len(rates))]
    loss = [i for i in range(len(rates))]
    for i,l_r in enumerate(rates):
        net = backprop_network.Network([784, 40, 10])
        print( "working on: ",l_r)
        train_accs, train_losses, test_accs = net.SGD_for_task_a(training_data, epochs=30, mini_batch_size=10, learning_rate=l_r, test_data=test_data)
        train[i],loss[i],test[i] = train_accs, train_losses, test_accs

    print("training is done!")
    plot_according_to_type("train accuracy",train,rates,"accuracy")
    plot_according_to_type("train loss",loss,rates,"loss")
    plot_according_to_type("test accuracy",test,rates,"accuracy")
    save_data(train,test,loss)

def plot_according_to_type(title,data,rates,y_label):
    epochs = [i for i in range(len(data[0]))]
    for i,rate in enumerate(rates):
        plt.plot(epochs,data[i],label=str(rate))
    plt.xlabel("epoch")
    plt.ylabel(y_label)
    leg = plt.legend(loc = 'center right')
    plt.title(title)
    plt.show()


def task_b():
    training_data, test_data = backprop_data.load(train_size=50000,test_size=10000)
    net = backprop_network.Network([784, 40, 10])
    net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1, test_data=test_data)

def bonus():
    training_data, test_data = backprop_data.load(train_size=50000,test_size=10000)
    net = backprop_network.Network([784, 500,10])
    net.SGD(training_data, epochs=100, mini_batch_size=10, learning_rate=0.14, test_data=test_data)

def find_best_learning_rate():
    training_data, test_data = backprop_data.load(train_size=10000,test_size=5000)
    net = backprop_network.Network([784, 40, 10])
    lears = np.arange(0.05,0.2,0.015)
    #mini_sizes = [10,15,20,25,30]
    for l in lears:
        print(l)
        net = backprop_network.Network([784, 40, 10])
        net.SGD_for_find_best(training_data, epochs=30, mini_batch_size=10, learning_rate=l, test_data=test_data)

def plot_according_to_type(title,data,rates,y_label):
    epochs = [i for i in range(len(data[0]))]
    for i,rate in enumerate(rates):
        plt.plot(epochs,data[i],label=str(rate))
    plt.xlabel("epoch")
    plt.ylabel(y_label)
    leg = plt.legend(loc = 'center right')
    plt.title(title)
    plt.show()


def save_data(train,test,loss):
    data = dict()
    data ["train accuracy"] = train
    data [ "test accuracy"] = test
    data["training loss"]= loss
    df = pd.DataFrame(data)
    df.to_csv("Backprop resuluts.csv")
#task_a(training_data, test_data)
#task_b()
bonus()
#find_best_learning_rate()
# plt.plot([0,1,2],[0,1,4])
# plt.show()
