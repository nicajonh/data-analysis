# -*- coding:utf-8 -*-
#锟竭硷拷锟截癸拷 锟皆讹拷锟斤拷模
import pandas as pd

#锟斤拷锟斤拷锟斤拷始锟斤拷
filename = '../data/bankloan.xls'
data = pd.read_excel(filename)
x = data.iloc[:,:8].as_matrix()
y = data.iloc[:,8].as_matrix()

from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import RandomizedLogisticRegression as RLR 
rlr = RLR() #锟斤拷锟斤拷锟斤拷锟斤拷呒锟斤拷毓锟侥ｏ拷停锟缴秆★拷锟斤拷锟�
rlr.fit(x, y) #训锟斤拷模锟斤拷
rlr.get_support() #锟斤拷取锟斤拷锟斤拷筛选锟斤拷锟斤拷锟揭诧拷锟斤拷锟酵拷锟�.scores_锟斤拷锟斤拷锟斤拷取锟斤拷锟斤拷锟斤拷锟斤拷锟侥凤拷锟斤拷
print(u'通锟斤拷锟斤拷锟斤拷呒锟斤拷毓锟侥ｏ拷锟缴秆★拷锟斤拷锟斤拷锟斤拷锟斤拷锟�')
print(u'锟斤拷效锟斤拷锟斤拷为锟斤拷%s' % ','.join(data.columns[rlr.get_support()]))
x = data[data.columns[rlr.get_support()]].as_matrix() #筛选锟斤拷锟斤拷锟斤拷

lr = LR() #锟斤拷锟斤拷锟竭硷拷锟斤拷锟斤拷模锟斤拷
lr.fit(x, y) #锟斤拷筛选锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟窖碉拷锟侥ｏ拷锟�
print(u'锟竭硷拷锟截癸拷模锟斤拷训锟斤拷锟斤拷锟斤拷锟斤拷')
print(u'模锟酵碉拷平锟斤拷锟斤拷确锟斤拷为锟斤拷%s' % lr.score(x, y)) #锟斤拷锟斤拷模锟酵碉拷平锟斤拷锟斤拷确锟绞ｏ拷锟斤拷锟斤拷为81.4%