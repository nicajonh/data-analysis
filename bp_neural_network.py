#-*- coding: utf-8 -*-
#基于bp网络的分类

import pandas as pd

#文件
inputfile = 'data/sales_data.xls'
data = pd.read_excel(inputfile, index_col = u'���') #��������

#数据转换:标签->数字，1-'好'、'是'、'高',0-反之
data[data=='高']=1
data[data=='是']=1
data[data=='好']=1

data[data != 1] = 0
x = data.iloc[:,:3].as_matrix().astype(int)
y = data.iloc[:,3].as_matrix().astype(int)

from keras.models import Sequential
from keras.layers.core import Dense, Activation

model = Sequential()  #建立模型
model.add(Dense(input_dim = 3, output_dim = 10))
model.add(Activation('relu')) #relu 激活器
model.add(Dense(input_dim = 10, output_dim = 1))
#由于是0-1输出，sigmoid函数作为激活函数
model.add(Activation('sigmoid')) #sigmoid激活器

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', class_mode = 'binary')

model.fit(x, y, nb_epoch = 2000, batch_size = 10) #训练两千次
yp = model.predict_classes(x).reshape(len(y)) 

from cm_plot import *
cm_plot(y,yp).show() 
