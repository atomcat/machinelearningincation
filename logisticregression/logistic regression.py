
# coding: utf-8

# 逻辑回归：多采用随机梯度下降法学习得到模型的最佳参数w（或者theta），本例中在学习过程中依据迭代次数来降低学习率，每次迭代学习，仅随机取一共样本参与学习。这种方法计算量小，需要内存小，速度快。

# In[99]:


import numpy as np
import pandas as pd
import os

# def loadData(file):
#     data = np.array(pd.read_csv(file))
#     newdata = np.c_[np.ones(data.shape[0]), data]
#     return newdata
#     n,m = data.shape
#     y = data[:, -1]
#     X = data[:, :-1]
#     return X, y, n, m-1

def loadDataSet():
    #准备数据和标签的空list
    dataMat = []
    labelMat = []
    fr = open('data/testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.array(dataMatIn)             #convert to NumPy array
    labelMatrix = np.array(classLabels)          #convert to NumPy array
    labelMatrix = labelMatrix.reshape((-1, 1))
    m, n = dataMatrix.shape
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1))
    for k in range(maxCycles):                       
        h = sigmoid(np.dot(dataMatrix, weights))     #calculate hw(x*w)
        error = (labelMatrix - h)                    #vector subtraction
        weights = weights + alpha * np.dot(dataMatrix.T, error) 
    return weights

def stocGradAscent0(dataMatIn, classLabels):
    dataMatrix = np.array(dataMatIn)             #convert to NumPy array
    labelMatrix = np.array(classLabels)          #convert to NumPy array
    labelMatrix = labelMatrix.reshape((-1, 1))
    m, n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)   #initialize to all ones
    #iterate m times, each time choose one sampe i to update weights
    for i in range(m):
        h = sigmoid(np.dot(dataMatrix[i], weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

def stocGradAscent1(dataMatIn, classLabels, numIter=150):
    dataMatrix = np.array(dataMatIn)             #convert to NumPy array
    labelMatrix = np.array(classLabels)          #convert to NumPy array
    labelMatrix = labelMatrix.reshape((-1, 1))
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)   #initialize to all ones
    
    #epoch numter times, each time, randomly choose a sample to update weights
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            #decrease learning rate with iteration, will not go to zero by 0.0001
            alpha = 4/(1.0+j+i)+0.0001    
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            h = sigmoid(np.dot(dataMatrix[randIndex],weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

def plotBestFit(weights):
    import matplotlib.pyplot as plt
    get_ipython().run_line_magic('matplotlib', 'inline')
    dataMatrix, labelMatrix = loadDataSet()
    dataArr = np.array(dataMatrix)
    n = np.shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMatrix[i]) == 1:
            xcord1.append(dataArr[i, 1]); ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1]); ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y.transpose())
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()
    
def classifyVector(inX, weights):
    #calculate the probability of input X
    prob = sigmoid(np.dot(inX, weights))
    if prob > 0.5: 
        return 1.0
    else: 
        return 0.0

def colicTest():
    #load training data and test data
    frTrain = open('data/horseColicTraining.txt'); frTest = open('data/horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split() #do not using split('\t'), this will contain tab, space, return
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    
    #using stochastic gradient ascent to learn weights, epoch 1000
    trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 1000)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        #split by tab ,space or return
        currLine = line.strip().split()
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate

def multiTest():
    numTests = 10; errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))
# def plotBestFit(w):
#     import matplotlib.pyplot as plt
#     data = loadData('data/testSet.csv')
#     pos = data[np.where(data[:,-1] == 1)]
#     neg = data[np.where(data[:,-1] == 0)]
    
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.scatter(pos[:,0], pos[:,1], s=30, c='red', marker='s')
#     ax.scatter(neg[:,0], neg[:,1], s=30, c='g', marker='s')
#     x = np.arange(-3.0, 3.0, 0.1)
#     y = (-w[0]-w[1]*x)/w[2]
#     ax.plot(x,y)
#     plt.xlabel('X1')
#     plt.ylabel('X2')
#     plt.show()


# data, label = loadDataSet()
# weights = gradAscent(data, label)
# plotBestFit(weights)
# weights = stocGradAscent0(data, label)
# plotBestFit(weights)
# weights = stocGradAscent1(data, label)
# plotBestFit(weights)

colicTest()
multiTest()

