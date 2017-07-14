
import numpy as np 
import pandas as pd


test = pd.read_csv('Test.csv')
train = pd.read_csv('Train.csv')
from sklearn.model_selection import train_test_split


X = train.drop('Item_Outlet_Sales',1);
x_train, x_cv, y_train, y_cv = train_test_split(X,train.Item_Outlet_Sales, test_size =0.3);

