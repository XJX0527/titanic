# -*- codeing = utf-8 -*-
# @Time:2021/5/31 21:34
# @Author:A20190277
# @File:tita.py
# @Software:PyCharm

import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

train=pd.read_csv(r"C:\Users\18356\Desktop\Kaggle\Titanic-Machine Learning from Disaster\titanic\train.csv")
print(train.shape)
test=pd.read_csv(r"C:\Users\18356\Desktop\Kaggle\Titanic-Machine Learning from Disaster\titanic\test.csv")
print(test.shape)
rowNum_train=train.shape[0]
rowNum_test=test.shape[0]
full=train.append(test,ignore_index=True)   #合并数据集
print(full.shape)   #便于对数据进行清洗
print(full.head())   #查看数据

print(full.describe()) #数据描述性统计分析
full.info()      #查看每一列的数据类型以及数据总数
print('-----------------------------------------------------------------------------')


'''数据清洗'''
print('处理数据之前')
full.info()
full['Age']=full['Age'].fillna(full['Age'].mean())   #平均值填充缺失值
full['Fare']=full['Fare'].fillna(full['Fare'].mean())
'''
随机森林填补缺失值
'''
print('数据处理之后')

full.info()
print(full.head())
#字符型缺失值处理完毕，处理字符串型缺失值
#登船舱口
full['Embarked'].head()
full['Embarked'].value_counts()   #将该变量分类，查看其中最常见登船舱口
full['Embarked']=full['Embarked'].fillna('S')   # 用最常见的登船舱口对缺失值进行填充
#由于船舱号的缺失值缺失数据较多，先用'U'将缺失值填充
full['Cabin']=full['Cabin'].fillna('U')
print(full.head())
full.info()
print('缺失值处理完毕！')

#数值类型数据：乘客编号PassengerId；年龄：Age；船票价格：Fare；直系亲属人数：SibSp；旁系亲属人数：Parch；
#时间序列：无
#分类数据：乘客性别Sex：男性：male；女性：female；
#登船港口Embarked：出发地1：S，途径地：C，出发地2：Q
#船舱等级Pclass：1=1等仓，2=2等仓，3=3等仓
#字符串类型：乘客姓名Name；客舱号：Cabin；船票编号：Ticket

print('处理性别----映射')
print(full['Sex'].head())
sex_mapDict={'male':1,'female':0}    #将male映射为1，female映射为：0
full['Sex']=full['Sex'].map(sex_mapDict)    #map()会根据提供的函数对指定序列做映射
print(full['Sex'].head())

print('处理登船港口----one-hot编码')
#存放提取后的特征
embarkedDf = pd.DataFrame()
embarkedDf = pd.get_dummies( full['Embarked'] , prefix='Embarked' )
print(embarkedDf.head())
#主要用于将分类变量进行one-hot的编码,参数 prefix就是前缀的的意思，是根据编码的向量名的前缀进行命名。

#添加one-hot编码产生的虚拟变量（dummy variables）到泰坦尼克号数据集full中，并将原有列Emarked删除
full = pd.concat([full,embarkedDf],axis=1)    #合并数据集
full.drop('Embarked',axis=1,inplace=True)     #删除列
print(full.head())

print('处理客舱等级---ont-hot编码')
pclassDf = pd.DataFrame()
pclassDf = pd.get_dummies( full['Pclass'] , prefix='Pclass' )
print(pclassDf.head())
full = pd.concat([full,pclassDf],axis=1)
full.drop('Pclass',axis=1,inplace=True)
print(full.head())

print('处理乘客姓名---由于姓名可能包含头衔（地位），所以定义函数来从Name中获取头衔')
def getTitle(name):
    str1=name.split(',')[1]
    str2=str1.split('.')[0]
    str3=str2.strip()
    return str3
titleDf=pd.DataFrame()
titleDf['Title']=full['Name'].map(getTitle)
print(titleDf.head())
print(Counter(titleDf['Title']))   #查看每个头衔的频数
#officer---政府官员；royalty---皇室；Mr---已婚男士；Mrs---已婚女士；Miss---未婚女士；Master---有技能的人等等
#建立映射关系
title_mapDict = {
                    "Mr" :        "Mr",
                    "Miss" :      "Miss",
                    "Mrs" :       "Mrs",
                    "Master" :    "Master",
                    "Rev":        "Officer",
                    "Dr":         "Officer",
                    "Col":        "Officer",
                    "Ms":         "Mrs",
                    "Major":      "Officer",
                    "Mlle":       "Miss",
                    "Don":        "Royalty",
                    "Mme":        "Mrs",
                    "Lady" :      "Royalty",
                    "Sir" :       "Royalty",
                    "Capt":       "Officer",
                    "the Countess":"Royalty",
                    "Jonkheer":   "Royalty",
                    "Dona":       "Royalty"
                    }
titleDf['Title']=titleDf['Title'].map(title_mapDict)
#ont-hot编码
titleDf=pd.get_dummies(titleDf['Title'])
print(titleDf.head())
full=pd.concat([full,titleDf],axis=1)
full.drop('Name',axis=1,inplace=True)
print(full.head())

print('处理客舱号---ont-hot')
print(Counter(full['Cabin']))
cabinDf=pd.DataFrame()
full['Cabin']=full['Cabin'].map(lambda c:c[0])
cabinDf=pd.get_dummies(full['Cabin'],prefix='Cabin')
print(cabinDf.head())
full=pd.concat([full,cabinDf],axis=1)
full.drop('Cabin',axis=1,inplace=True)
print(full.head())

print('处理家庭人数=直系+旁系+1')
#将人数分类
familyDf=pd.DataFrame()
familyDf['FamilySize']=full['Parch']+full['SibSp']+1
print(familyDf['FamilySize'].describe())
#人数=1：小家庭；人数[2,4]：中等家庭；人数>=5：大家庭
#运用one-hot思路，对数据进行映射处理
familyDf['Family_Single']=familyDf['FamilySize'].map(lambda s:1 if s==1 else 0)
familyDf['Family_Small']=familyDf['FamilySize'].map(lambda s:1 if 2<=s<=4 else 0)
familyDf['Family_Large']=familyDf['FamilySize'].map(lambda s:1 if s>=5 else 0)
print(familyDf.head())
full=pd.concat([full,familyDf],axis=1)
print(full.head())
#保存一下数据
#full.to_csv(r'C:\Users\18356\Desktop\Kaggle\Titanic-Machine Learning from Disaster\titanic\full.csv',index=False)
print('数据处理完毕！选择特征值')
corrDf=full.corr()
#print(corrDf)
#查看各个特征与生成情况（Survived）的相关系数，ascending=False表示按降序排列
print(corrDf['Survived'].sort_values(ascending=False))
#根据相关系数以及生活经验
#我们剔除以下数据：ID、票编号
full_x=pd.concat([titleDf,pclassDf,familyDf,full['Fare'],cabinDf,embarkedDf,full['Sex'],full['Age']],axis=1)
print(full_x.head())

print('构建模型！---利用train数据进行构建')
sourceRow=train.shape[0]
print(sourceRow)
source_x=full_x.loc[0:sourceRow-1,:]    #特征（原始）
source_y=full.loc[0:sourceRow-1,'Survived']   #标签
pred_x=full_x.loc[sourceRow:,:]      #特征（预测）

print('原始数据',source_x.shape[0])
print('预测数据',source_y.shape[0])

#从full数据中拆分出train与test
from sklearn.model_selection import train_test_split
#将train拆分为训练集和测试集
train_x,test_x,train_y,test_y=train_test_split(source_x,source_y,train_size=.8)
print('原始数据',source_x.shape,'训练集特征',train_x.shape,'测试集特征',test_x.shape)
print('原始数据',source_y.shape,'训练集特征',train_y.shape,'测试集特征',test_y.shape)

#选择算法
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
#随机森林算法
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100)

'''
#Support Vector Machines算法
from sklearn.svm import SVC,LinearSVC
model=SVC()
#Gradient Boosting Classifier算法
from sklearn.ensemble import GradientBoostingClassifier
model=GradientBoostingClassifier()
#K-nearest neighbors算法
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=3)
#Gaussian Naive Bayes算法
from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
'''
print('训练模型！')
model.fit(train_x,train_y)
print(model.fit(train_x,train_y))

print('评估模型！')
model.score(test_x,test_y)
print(model.score(test_x,test_y))
pred_y=model.predict(pred_x)
pred_y= pred_y.astype(int)
passenger_id=full.loc[sourceRow:,'PassengerId']
predDf=pd.DataFrame({'PassengerId':passenger_id,'Survived':pred_y})
predDf.shape
print(predDf.head())
#predDf.to_csv(r'C:\Users\18356\Desktop\Kaggle\Titanic-Machine Learning from Disaster\titanic\pred.csv',index=False)