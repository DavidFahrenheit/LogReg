
import numpy as np
import pandas as pd
import sklearn
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression

scaler = preprocessing.MinMaxScaler()


Bel_dat = pd.read_csv('C:\\Users\\Rob\\Documents\\VSCPython\\Covid19\\Belgium.txt', sep ="\s+")
Fra_dat = pd.read_csv('C:\\Users\\Rob\\Documents\\VSCPython\\Covid19\\France.txt', sep ="\s+")
Ger_dat = pd.read_csv('C:\\Users\\Rob\\Documents\\VSCPython\\Covid19\\Germany.txt', sep ="\s+")
Ita_dat = pd.read_csv('C:\\Users\\Rob\\Documents\\VSCPython\\Covid19\\Italy.txt', sep ="\s+")
Ned_dat = pd.read_csv('C:\\Users\\Rob\\Documents\\VSCPython\\Covid19\\Netherlands.txt', sep ="\s+")
Nzl_dat = pd.read_csv('C:\\Users\\Rob\\Documents\\VSCPython\\Covid19\\New Zealand.txt', sep ="\s+")
Spa_dat = pd.read_csv('C:\\Users\\Rob\\Documents\\VSCPython\\Covid19\\Spain.txt', sep ="\s+")
Swe_dat = pd.read_csv('C:\\Users\\Rob\\Documents\\VSCPython\\Covid19\\Sweden.txt', sep ="\s+")
Usa_dat = pd.read_csv('C:\\Users\\Rob\\Documents\\VSCPython\\Covid19\\United States.txt', sep ="\s+")


Bel_rows, Bel_cols = Bel_dat.shape
Fra_rows, Fra_cols = Fra_dat.shape
Ger_rows, Ger_cols = Ger_dat.shape
Ita_rows, Ita_cols = Ita_dat.shape
Ned_rows, Ned_cols = Ned_dat.shape
Nzl_rows, Nzl_cols = Nzl_dat.shape
Spa_rows, Spa_cols = Spa_dat.shape
Swe_rows, Swe_cols = Swe_dat.shape
Usa_rows, Usa_cols = Usa_dat.shape

Bel_start = 0
Bel_end = Bel_rows
Fra_start = Bel_end
Fra_end = Fra_start + Fra_rows
Ger_start = Fra_end
Ger_end = Ger_start + Ger_rows
Ita_start = Ger_end
Ita_end = Ita_start + Ita_rows

#print("Belgium rows = ", Bel_rows)

CV_data = pd.concat((Bel_dat,Fra_dat,Ger_dat,Ita_dat,Ned_dat,Nzl_dat,Spa_dat,Swe_dat,Usa_dat),axis=0)


#code below doesn't work 
#CV_data.drop(columns = 'Date')

X = CV_data.iloc[:,0:-1]
Y = CV_data.iloc[:,-1]

Xnp = X.to_numpy()
Ynp = Y.to_numpy()

X0 = np.multiply(Xnp[0,:],Xnp)
X1 = np.multiply(Xnp[1,:],Xnp)
X2 = np.multiply(Xnp[2,:],Xnp)
X3 = np.multiply(Xnp[3,:],Xnp)
X4 = np.multiply(Xnp[4,:],Xnp)
X_mult = np.concatenate((Xnp,X0,X1,X2,X3,X4), axis=1)

#Xnp = np.delete(Xnp1,1,axis =1)

X_sc = scaler.fit_transform(X_mult)

#print(X_sc)

X2 = scaler.fit_transform(Xnp**2)
X3 = scaler.fit_transform(Xnp**3)

Xtanh= (np.tanh((X_sc-0.2)/2)+1-(np.tanh((X_sc-0.4)/2)+1))
X_xxx = np.multiply(X_sc[1,:], Xtanh)
X_cos = np.cos(X_sc)
X_sin = np.sin(X_sc)
XinvLog = scaler.fit_transform(np.log((Xnp+0.1)**(-1)))

X_train = np.concatenate((X_sc,Xtanh), axis =1)
#X_train = X_sc

print('X_train shape = ', X_train.shape)

linmodel = LinearRegression()

linmodel.fit(X_train, Ynp)

Mtheta = linmodel.coef_ 
Mintercept = linmodel.intercept_

y_pred = linmodel.predict(X_train)	

#print(Ynp)
#print(y_pred)

#print(Mtheta)
#print(Mintercept)

#np.savetxt('2darray.csv', X_train, delimiter=',', fmt='%d')
#np.savetxt('thetas.csv', Mtheta, delimiter=',', fmt='%d')

#print(Xnp[:,1].shape)
plt.subplot(321)
plt.plot(Xnp[Bel_start:Bel_end,0],y_pred[Bel_start:Bel_end], label = "predicted")
plt.plot(Xnp[Bel_start:Bel_end,0],Ynp[Bel_start:Bel_end], label = "actual")
plt.xlabel('days after 5th case')
plt.ylabel('deaths')
plt.title('Belgium')

plt.subplot(322)
plt.plot(Xnp[Fra_start:Fra_end,0],y_pred[Fra_start:Fra_end], label = "predicted")
plt.plot(Xnp[Fra_start:Fra_end,0],Ynp[Fra_start:Fra_end], label = "actual")
plt.xlabel('days after 5th case')
plt.ylabel('deaths')
plt.title('France')

plt.subplot(323)
plt.plot(Xnp[Ger_start:Ger_end,0],y_pred[Ger_start:Ger_end], label = "predicted")
plt.plot(Xnp[Ger_start:Ger_end,0],Ynp[Ger_start:Ger_end], label = "actual")
plt.xlabel('days after 5th case')
plt.ylabel('deaths')
plt.title('Germany')

plt.subplot(324)
plt.plot(Xnp[Ita_start:Ita_end,0],y_pred[Ita_start:Ita_end], label = "predicted")
plt.plot(Xnp[Ita_start:Ita_end,0],Ynp[Ita_start:Ita_end], label = "actual")
plt.xlabel('days after 5th case')
plt.ylabel('deaths')
plt.title('Italy')

plt.show()