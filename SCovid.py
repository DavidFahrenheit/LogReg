
import numpy as np
import pandas as pd
import sklearn
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression

scaler = preprocessing.MinMaxScaler()


CV_Data = pd.read_csv('CovidData.txt', sep ="\s+")

#print(CV_Data.shape)

X = CV_Data.iloc[:,0:-1]
Y = CV_Data.iloc[:,-1]

Xnp = X.to_numpy()
Ynp = Y.to_numpy()

X_sc = scaler.fit_transform(Xnp)

#print(X_sc)

X2 = X_sc*X_sc
X3 = X2*X_sc

Xtanh= np.tanh(X_sc)
X_cos = np.cos(X_sc)
X_sin = np.sin(X_sc)

X_train = np.concatenate((X_sc,X2,X3,Xtanh,X_cos,X_sin), axis =1)

linmodel = LinearRegression 

linmodel.fit(X_train, Ynp)

Mtheta = linmodel.coef_ 
Mintercept = linmodel.intercept_

y_pred = linmodel.predict(X_train)	

#print(Ynp)
print(y_pred)

#print(Mtheta)
#print(Mintercept)

#np.savetxt('2darray.csv', X_train, delimiter=',', fmt='%d')
#np.savetxt('thetas.csv', Mtheta, delimiter=',', fmt='%d')

#plt.scatter(Ynp,y_pred)

#plt.savefig('plot2')
