## import libraries 

import numpy as np 
import pandas as pd

from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

### import test and train file

test = pd.read_csv('Test.csv')
train = pd.read_csv('Train.csv')

## data imputation 

train['Item_Weight'].fillna((train['Item_Weight'].mean()), inplace=True)

leg = LinearRegression()

## splotting data into training and CV for cross validation 

X = train.loc[:, ['Outlet_Establishment_Year', 'Item_MRP', 'Item_Weight']]

x_train, x_cv, y_train, y_cv = train_test_split(X,train.Item_Outlet_Sales)

## x_train, x_cv , y_train, y_cv = train_test_split(X, )
##print len(y_train), len(x_train), len(x_cv), len(y_cv);

leg.fit(x_train, y_train)

## predicting on cv 

pred = leg.predict(x_cv)


## calculating mse 
mse = np.mean((pred - y_cv)**2)
print 'this is the result ' 
print mse
print type(train)