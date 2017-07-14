import numpy as np 
import pandas as pd

from data_import import *
# test = pd.read_csv('Test.csv')
# train = pd.read_csv('Train.csv')
from sklearn.model_selection import train_test_split

## imputing missing values

##print train['Item_Visibility'];

#print x_train.isnull().sum()

train['Item_Visibility'] = train['Item_Visibility'].replace(0, np.mean(train['Item_Visibility']))

print (train['Outlet_Establishment_Year']).value_counts()
## print np.in1d(train['Item_Visibility'], 0).sum()
