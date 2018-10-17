
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd

def loadData(file):
    data = np.array(pd.read_csv(file))
    newdata = np.c_[np.ones(data.shape[0]), data]
    return newdata
#     n,m = data.shape
#     y = data[:, -1]
#     X = data[:, :-1]
#     return X, y, n, m-1

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def gradAscent(file):
    X, y, n, m = loadData(file)
    print (y)
    print (X)
    print (data.shape)
    alpha = 0.001
    epoch = 500
    w = np.ones((m-1,1))
    for k in range(epoch):
        h = sigmoid(np.dot(X, w))
        error = (y - h)
        w = w + alpha * np.dot(X.T, error)
    return w

def plotBestFit(w):
    import matplotlib.pyplot as plt
    data = loadData('../data/data.csv')
    pos = data[np.where(data[:,-1] == 1)]
    neg = data[np.where(data[:,-1] == 0)]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(pos[:,0], pos[:,1], s=30, c='red', marker='s')
    ax.scatter(neg[:,0], neg[:,1], s=30, c='g', marker='s')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-w[0]-w[1]*x)/w[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def stocGradAscent(data):
    m,n = data.shape
    print(m, n)
    alpha = 0.01
    weights = np.ones(n-1)
    for i in range(m):
        h = sigmoid(np.dot(data[i,:-1], weights))
        error = data[i, -1] - h
        weights = weights +alpha * error * data[i, :-1]
    return weights

def stocGradAscent1(data, epoch=150):
    m, n = data.shape
    w = np.ones(n-1)
    for j in range(epoch):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4 / (1.0 +j + i) + 0.01
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            h = sigmoid(np.dot(data[randIndex, :-1], w))
            error = data[randomIndex, -1] - h
            w = w + alpha * np.dot(data[randIndex, :-1], error)
            del(dataIndex[randIndex])
    return w

def classifyVector(X, weights):
    prob = sigmoid(np.dot(X, weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

    

data = loadData('data/horseColicTraning.csv')
print (data)

