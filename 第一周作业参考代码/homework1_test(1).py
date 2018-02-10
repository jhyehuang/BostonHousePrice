import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,OneHotEncoder

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import minmax_scale
from scipy.sparse import hstack

data = pd.read_csv('day.csv')
data = data.drop(['instant','registered','dteday','casual'],axis=1)

OneHotEnc = OneHotEncoder()
mn_x = MinMaxScaler()

#选取2011年的数据作为训练数据
train_data = data[data.yr ==0]

#类别型变量进行OneHot编码
x_train_cat= OneHotEnc.fit_transform(train_data[['season','mnth','holiday','weekday','workingday']])

#数据值变量继续预处理
x_train_num = mn_x.fit_transform(train_data[['temp','atemp','hum','windspeed']])

#将变换后的类别型变量和数值型变量串联
x_train = hstack((x_train_cat,x_train_num))

y_train = train_data['cnt'].values
print("shape of x train:", x_train.shape)
print("shape of y train",y_train.shape)

#选择2012年的数据为测试数据
test_data = data[data.yr == 1]

#测试数据进行与训练数据一样的预处理（直接transform，没有fit了，用训练数据fit）
x_test_cat = OneHotEnc.transform(test_data[['season','mnth','holiday','weekday','workingday']])
x_test_num = mn_x.transform(test_data[['temp','atemp','hum','windspeed']])
x_test = hstack((x_test_cat,x_test_num))
y_test = test_data['cnt'].values
print("shape of x test:", x_test.shape)
print("shape of y test:",y_test.shape)

#对y做标准化，不是必须
ss_y = StandardScaler()
y_train = ss_y.fit_transform(y_train.reshape(-1,1))
y_test = ss_y.transform(y_test.reshape(-1,1))

#训练集和测试集y的均值差异很大，均值差异用作校正
mean_test_y = y_test.mean()
#归一化后train均值为0
#mean_train_y = 0
mean_diff = mean_test_y;
print("difference between mean of train and test y:", mean_diff)

#1. 缺省参数的最小二乘
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
y_test_pred_lr = lr.predict(x_test)
y_test_pred_lr += mean_diff
y_train_pred_lr = lr.predict(x_train)

print('The r2 score of LinearRegression on test is',r2_score(y_test,y_test_pred_lr))
print('The r2 score of LinearRegression on train is',r2_score(y_train,y_train_pred_lr))

#2. 岭回归，对正则参数lambda（scikit learn中为alpha）进行调优
from sklearn.linear_model import  RidgeCV

#设置超参数（正则参数）范围
alphas = [ 0.01, 0.1, 1, 10,100]
#n_alphas = 20
#alphas = np.logspace(-5,2,n_alphas)

#生成一个RidgeCV实例
ridge = RidgeCV(alphas=alphas)

#模型训练
ridge.fit(x_train, y_train)

#预测
y_test_pred_ridge = ridge.predict(x_test)
y_test_pred_ridge += mean_diff
y_train_pred_ridge = ridge.predict(x_train)


# 评估，使用r2_score评价模型在测试集和训练集上的性能
print (('The r2 score of RidgeCV on test is'), r2_score(y_test, y_test_pred_ridge))
print (('The r2 score of RidgeCV on train is'), r2_score(y_train, y_train_pred_ridge))

#3. Lasso，对正则参数lambda（scikit learn中为alpha）进行调优
#过程与岭回归一样，在此省略