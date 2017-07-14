



import data_import
from sklearn.linear_model import Ridge

## training the model 

ridgeReg = Ridge(alpha = 0.05, normalize = True)
ridgeReg.fit(x_train, y_train)

pred_rid = ridgeReg.predict(x_cv)

##calculating mse

mse = np.mean((pred_rid - y_cv) **2)

print mse;