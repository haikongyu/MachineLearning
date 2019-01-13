# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 10:35:35 2018

@author: Administrator
"""

def getminindex(lis):
    '''
    该函数的功能是找到列表lis中最小值的索引
    '''
    
    #注意使用浅复制拷贝原始列表
    
    lis_copy = lis[:]
    lis_copy.sort()
    minvalue = lis_copy[0]
    minindex = lis.index(minvalue)
    return minindex

from sklearn.preprocessing import scale
from scipy.spatial.distance import mahalanobis 
import pandas as pd
import numpy as np

def mahalanuobis_discrim(x_test , x_train , train_label):
    '''
    该函数用于使用马氏距离进行判别分析 ，各参数的含义如下：
    x_test:DataFrame 类型的待判别样本 ，为k 行 m列（k 为待判别样本个数 ，m为变量个数）
    x_train:DataFrame 类型的训练数据 ， 训练数据每个变量均为数值型 ， 不含标签值。为n行
            m列（n 为训练样本个数 ， m为变量个数）
    train_label: 训练样本中每个样本对应的标签 ，应当为n行1列
    '''
    #将最终判别结果存入列表 final_result中
    final_result = []
    #变量名
    colname =  x_train.columns
    #test_n , train_n ,和m分别用于存储待判别样本个数，训练样本个数，变量个数
    test_n = x_test.shape[0]
    train_n = x_train.shape[0]
    m = x_train.shape[1]
    n = test_n + train_n
    #data_x存储训练数据和测试数据组成的自变量数据，Data_x_scale存储标准化自变量数据
    data_x = x_train.append(x_test)
    data_x_scale = scale(data_x)
    x_train_scale = pd.DataFrame(data_x_scale[:train_n])
    x_test_scale = pd.DataFrame(data_x_scale[train_n:])
    #将train_label和xx_train合并组成训练集 ，用于按照label类别求出各类中心
    data_train = x_train_scale.join(train_label)
    #miu用于存储各类别中心，lanel_name用于存储训练数据标签的列名
    label_name = data_train.columns[-1]
    miu = data_train.groupby(label_name).mean()
    miu = np.array(miu)
    print('类中心： ')
    print(pd.DataFrame(miu))
    print()
    #将标签存储于列表 label中
    label = train_label.drop_duplicates()
    label = label.iloc[: , 0]
    label = list(label)
    label_len = len(label)
    #将数据转换为numpy中的array ,方便后续操作
    x_test_array = np.array(x_test_scale)
    x_train_array = np.array(x_train_scale)
    data_x_scale_array = np.array(data_x_scale)
    #计算协方差矩阵
    cov = np.cov(data_x_scale_array.T)
    #计算训练样本和测试样本到各个类中心的马氏距离并由此将其归类
    for i in range (n):
        dist = []
        for j in range(label_len):
            d = float(mahalanobis(data_x_scale[i] , miu[j] , np.mat(cov).I))
            dist.append(d)
        min_dist_index = getminindex(dist)
        #将样本到类中心的距离对应到类别上，得到判别结果，存入 result中
        result = label[min_dist_index]
        final_result.append(result)
    print('分类结果为： ')
    return final_result
