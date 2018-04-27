## import libraries 

import numpy as np 
import pandas as pd
import data_import
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from pylab import * 

### import test and train file

#test = pd.read_csv('Test.csv')
#train = pd.read_csv('Train.csv')

leg = LinearRegression()

## splotting data into training and CV for cross validation 

#X = train.loc[:, ['Outlet_Establishment_Year', 'Item_MRP']]

#x_train, x_cv, y_train, y_cv = train_test_split(X,train.Item_Outlet_Sales)

## x_train, x_cv , y_train, y_cv = train_test_split(X, )
##print len(y_train), len(x_train), len(x_cv), len(y_cv);

leg.fit(x_train, y_train)

## predicting on cv 

pred = leg.predict(x_cv)


## calculating mse 
mse = np.mean((pred - y_cv)**2)
##print 'this is the result ' 
##print mse

##### calculating coefficients 

#coeff = DataFrame(x_train.columns)
#coeff['Coefficients Estimates'] = Series(lreg.coef_)
## print Series(leg.coef_);

### checking the magnitude of cofficients

predictors = x_train.columns

coef = Series(leg.coef_, predictors).sort_values()
print coef.plot(kind= 'bar', title = 'Model');
print('done2')

