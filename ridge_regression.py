



from data_import import *
from sklearn.linear_model import Ridge;
from sklearn.grid_search import RandomizedSearchCV
from sklearn.grid_search import GridSearchCV


## training the model 

ridgeReg = Ridge(alpha = 0.05, normalize = True)
ridgeReg.fit(x_train, y_train)

pred_rid = ridgeReg.predict(x_cv)

##calculating mse

mse = np.mean((pred_rid - y_cv) **2)

print mse;