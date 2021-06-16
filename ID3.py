# -*- codeing = utf-8 -*-
# @Time:2021/6/16 18:06
# @Author:A20190277
# @File:ID3.py
# @Software:PyCharm

#ID3算法---根据信息熵来选择变量，分类树
from math import log
import operator

def calcShannonEnt(dataSet):  #定义函数---计算数据的熵
    numEhtries=len(dataSet)   #数据条数
    labelCounts={}            #标签
    shannonEnt=0             #初始化熵值
    for featVec in dataSet:
        currentLabel=featVec[-1]   #每个数据的最后一个元素
        if currentLabel not in labelCounts.keys():  #判断特征值是否为空
            labelCounts[currentLabel]=0    #特征值为空，返回单节点树
        labelCounts[currentLabel]+=1       #labelCounts是一个字典，统计有多少类别以及每个类别中的数量
    for key in labelCounts:
        prob=float(labelCounts[key])/numEhtries   #每个类别的熵值
        shannonEnt=prob*log(prob,2)   #计算熵值的和
    return shannonEnt

def createDataSet1():    # 创造示例数据
    dataSet = [['长', '粗', '男'],
               ['短', '粗', '男'],
               ['短', '粗', '男'],
               ['长', '细', '女'],
               ['短', '细', '女'],
               ['短', '粗', '女'],
               ['长', '粗', '女'],
               ['长', '粗', '女']]
    labels = ['头发','声音']  #两个特征
    return dataSet,labels

def createDataSet2():    # 创造示例数据
    dataSet = [['长', '粗', '男'],
               ['短', '粗', '男'],
               ['短', '粗', '男'],
               ['长', '细', '女'],
               ['短', '细', '女'],
               ['短', '粗', '女'],
               ['长', '粗', '女'],
               ['长', '粗', '女']]
    labels = ['声音','头发']  #两个特征
    return dataSet,labels

def splitDataSet(dataSet,axis,value):  #按某个特征分类之后的数据
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis]==value:
            reducedFeatVec=featVec[:axis]
            reducedFeatVec.extent(featVec[axis+1:])   #添加数据
            retDataSet.append(reducedFeatVec)         #合并数据
    return retDataSet

def chooseBestFeatureToSplit(dataSet):     #选择最优特征
    numFeatures=len(dataSet[0])-1
    baseEntropy=calcShannonEnt(dataSet)    #原始熵值
    bastInfoGain=0
    bestFeature=-1
    for i in range(numFeatures):
        featList=[example[i] for example in dataSet]
        uniqueVals=set(featList)
        newEntropy=0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)   #按特征分类后的熵
        infoGain = baseEntropy - newEntropy      #原始熵与按特征分类后的熵的差值
        if (infoGain > bestInfoGain):           #若按某特征划分后，熵值减少的最大，则次特征为最优分类特征
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):   #分类后类别数量排序，Eg.最后按标签分类为2男1女，则判定为男
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    soretedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)   #sorted函数的用法
    return soretedClassCount[0][0]

def createTree(dataSet,labels):  #决策树函数
    classList=[example[-1] for example in dataSet]
    if classList.count(classList[0])==len(classList):
        return classList[0]
    if len(dataSet[0])==1:
        return majorityCnt(classList)
    bestFeat=chooseBestFeatureToSplit(dataSet) #选择最优特征
    bestFeatLabel=labels[bestFeat]
    myTree={bestFeatLabel:{}} #分类结果以字典形式保存
    del(labels[bestFeat])
    featValues=[example[bestFeat] for example in dataSet]
    uniqueVals=set(featValues)
    for value in uniqueVals:
        subLabels=labels[:]
        myTree[bestFeatLabel][value]=createTree(splitDataSet\
                            (dataSet,bestFeat,value),subLabels)
    return myTree


if __name__=='__main__':
    dataSet, labels=createDataSet1()  # 创造示列数据
    print(createTree(dataSet, labels))  # 输出决策树模型结果
    dataSet, labels = createDataSet2()  # 创造示列数据
    print(createTree(dataSet, labels))  # 输出决策树模型结果

