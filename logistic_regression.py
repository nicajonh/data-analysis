# -*- coding:utf-8 -*-
import pandas as pd
filename = '../data/bankloan.xls'
data = pd.read_excel(filename)
x = data.iloc[:,:8].as_matrix()
y = data.iloc[:,8].as_matrix()

from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import RandomizedLogisticRegression as RLR 
rlr = RLR() #建立随机逻辑回归模型,复筛选变量
rlr.fit(x, y) #训练模型
rlr.get_support() #获取特征筛选变量
print(u'有效特征为:%s' % ','.join(data.columns[rlr.get_support()]))
x = data[data.columns[rlr.get_support()]].as_matrix() #筛选锟斤拷锟斤拷锟斤拷

lr = LR() #建立逻辑回归模型
lr.fit(x, y) #训练模型
print(u'模型的平均正确率:%s' % lr.score(x, y))