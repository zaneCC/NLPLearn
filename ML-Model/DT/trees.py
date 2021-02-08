import numpy as np
import pandas as pd
from math import log
'''
    决策树轮子
'''

# 建立模拟数据
def createDataSet():
    dataSet = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet, labels

# 计算信息熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1] # 获取标记
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    h = 0.0
    for key in labelCounts:
        p = float(labelCounts[key]) / numEntries
        h -= p * log(p,2)

    return h

#划分数据集
def splitDataSet(dataset,axis,value):
    retdataset=[]#创建返回的数据集列表
    for featVec in dataset:#抽取符合划分特征的值
        if featVec[axis]==value:
            reducedfeatVec=featVec[:axis] #去掉axis特征
            reducedfeatVec.extend(featVec[axis+1:])#将符合条件的特征添加到返回的数据集列表
            retdataset.append(reducedfeatVec)
    return retdataset

# 选择最佳特征
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            p = len(subDataSet) / float(len(dataSet))
            newEntropy += p * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestFeature = i
            bestInfoGain = infoGain
    return bestFeature

def majorityCnt(classList):
    '''
    数据集已经处理了所有属性，但是类标签依然不是唯一的，
    此时我们需要决定如何定义该叶子节点，在这种情况下，我们通常会采用多数表决的方法决定该叶子节点的分类
    '''
    classCont={}
    for vote in classList:
        if vote not in classCont.keys():
            classCont[vote]=0
        classCont[vote]+=1
    sortedClassCont=sorted(classCont.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCont[0][0]

# 创建决策树
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]

    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels= labels[:]
        s = splitDataSet(dataSet, bestFeat, value)
        myTree[bestFeatLabel][value] = createTree(s, subLabels) # 建立子决策树
    return myTree

dataSet, labels = createDataSet()
r = createTree(dataSet,labels)
print(r)
