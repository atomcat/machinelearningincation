
# coding: utf-8

# In[9]:


from math import log
import numpy as np

def create_data_set():
    dataset = [[1, 1, 'yes'], 
               [1, 1, 'yes'],
               [1, 0, 'no'], 
               [0, 1, 'no'], 
               [0, 1, 'no']]
    labels = ['no surface', 'flippers']
    return dataset, labels

def shannon_entropy(dataset):
    n = len(dataset)
    labelcounts = {}
    
    #统计每一种label的个数
    for feature in dataset:
        current_label = feature[-1]
        if current_label not in labelcounts.keys():
            labelcounts[current_label] = 0
        labelcounts[current_label] += 1
        
    #将每一种label按信息熵公式计算并求和
    entropy = 0.0
    for key in labelcounts:
        prop = float(labelcounts[key]) / n
        entropy -= prop * log(prop, 2)
    return entropy

#对某个属性（axis），按属性的值将原数据划分得到一个子集数据
#要将属性本身的值去掉，取改值下其他所有属性和标签值
def split_data(dataset, axis, value):
    restdataset = []
    for feature in dataset:     
        if feature[axis] == value:
            reduced = feature[:axis]
            reduced.extend(feature[axis+1:])
            restdataset.append(reduced)
#     print("split++++++++++++++")
#     print(restdataset)
    return restdataset
        
def choose_best_feature_to_split(dataset):
    #获取属性个数
    feature_number = len(dataset[0]) -1
    base_entropy = shannon_entropy(dataset)
    best_info_gain = 0.0
    best_feature = -1
    
    #每个属性计算一次信息增益
    for i in range(feature_number):
        #取该属性所有值放入list
        feature_list = [example[i] for example in dataset]
        print (feature_list)
        #唯一化属性值
        uniquevals = set(feature_list)
        feature_entropy = 0.0
        #对每个值，划分数据后，计算信息熵
        for val in uniquevals:
            subdataset = split_data(dataset, i, val)
            prob = len(subdataset) / float(len(dataset))
            feature_entropy += prob * shannon_entropy(subdataset)
        #计算信息增益，父信息熵 - 按改属性分裂后信息熵
        info_gain = base_entropy - feature_entropy
        
        #获取大的信息增益及属性
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature

def majority_cnt(classlist):
    class_count = {}
    #按class 统计每一种的个数，降序排列，取第一个即票数最多那类的数量
    for vote in classlist:
        if vote not in class_count:
            class_count[vote] = 0
        class_count[vote] += 1
        sorted_class_count = sorted(class_count.iteritems(), 
                             key=operator.itemgetter(1), 
                             reverse=True)
    return sorted_class_count[0][0]

def create_tree(dataset, labels):
    #获取标签列表
    classlist = [example[-1] for example in dataset]
    #所有class labels相同，停止递归，
    if classlist.count(classlist[0]) == len(classlist):
        return classlist[0]
    #没有属性用于split了，停止递归，投票决定
    if len(dataset) == 1:
        return majority_cnt(classlist)
    
    #找出split的feature
    bestfeature = choose_best_feature_to_split(dataset)
    bestfeaturelabel = labels[bestfeature]
    
    #创建空树
    myTree = {bestfeaturelabel:{}}
    #从标签列表去掉最佳属性的位置
    del(labels[bestfeature])
    featurevalues = [example[bestfeature] for example in dataset]
    uniquevals = set(featurevalues)
    #遍历属性的每一种值，递归建树
    for val in uniquevals:
        sublabels = labels[:]
        myTree[bestfeaturelabel][val] = create_tree(split_data(dataset, bestfeature, val), sublabels)
    return myTree
   
def classify(inputtree, featurelabels, testvector):
    firstSides = list(inputtree.keys())
    first_str = firstSides[0]
    print(featurelabels)
    second_dict = inputtree[first_str]
    feature_index = featurelabels.index(first_str)
    #遍历树中所有key，如果与待检测向量的值相同，看看不是dict类型，是表明未找到，继续找，否则返回类型
    for key in second_dict.keys():
        if testvector[feature_index] == key:
            if type(second_dict[key]).__name__=='dict':
                classlabel = classify(seconddict[key], featurelabels, testvector)
            else:
                classlabel = second_dict[key]
    return classlabel
  
def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)

# dataset,labels = create_data_set()
# print(labels)
# orig_labels = labels.copy()

# split_data(dataset, 1, 1)
# choose_best_feature_to_split(dataset)
# myTree = create_tree(dataset, labels)
# print(orig_labels)
# classify(myTree,orig_labels,[1,1])
# shannon_entropy(dataset)


fr=open('../data/lenses.txt')
lenses=[inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels=['age', 'prescript', 'astigmatic', 'tearRate']
lensesTree = create_tree(lenses,lensesLabels)
lensesTree

