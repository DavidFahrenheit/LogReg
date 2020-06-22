
import numpy as np
import pandas as pd
import sklearn
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression

scaler = preprocessing.MinMaxScaler()

#Import data files by country
Bel_dat = pd.read_csv('C:\\Users\\robev\\Documents\\VSCode RME\\Belgium.txt', sep ="\s+")
Fra_dat = pd.read_csv('C:\\Users\\robev\\Documents\\VSCode RME\\France.txt', sep ="\s+")
Ger_dat = pd.read_csv('C:\\Users\\robev\\Documents\\VSCode RME\\Germany.txt', sep ="\s+")
Ita_dat = pd.read_csv('C:\\Users\\robev\\Documents\\VSCode RME\\Italy.txt', sep ="\s+")
Ned_dat = pd.read_csv('C:\\Users\\robev\\Documents\\VSCode RME\\Netherlands.txt', sep ="\s+")
Nzl_dat = pd.read_csv('C:\\Users\\robev\\Documents\\VSCode RME\\New Zealand.txt', sep ="\s+")
Spa_dat = pd.read_csv('C:\\Users\\robev\\Documents\\VSCode RME\\Spain.txt', sep ="\s+")
Swe_dat = pd.read_csv('C:\\Users\\robev\\Documents\\VSCode RME\\Sweden.txt', sep ="\s+")
Usa_dat = pd.read_csv('C:\\Users\\robev\\Documents\\VSCode RME\\United States.txt', sep ="\s+")

#Define number of rows and cols in each dataframe
Bel_rows, Bel_cols = Bel_dat.shape
Fra_rows, Fra_cols = Fra_dat.shape
Ger_rows, Ger_cols = Ger_dat.shape
Ita_rows, Ita_cols = Ita_dat.shape
Ned_rows, Ned_cols = Ned_dat.shape
Nzl_rows, Nzl_cols = Nzl_dat.shape
Spa_rows, Spa_cols = Spa_dat.shape
Swe_rows, Swe_cols = Swe_dat.shape
Usa_rows, Usa_cols = Usa_dat.shape

#Define start and end rows for each country, for when dataframes are concatenated into one dataframe
Bel_start = 0
Bel_end = Bel_rows
Fra_start = Bel_end
Fra_end = Fra_start + Fra_rows
Ger_start = Fra_end
Ger_end = Ger_start + Ger_rows
Ita_start = Ger_end
Ita_end = Ita_start + Ita_rows
Ned_start = Ita_end
Ned_end = Ned_start + Ned_rows
Nzl_start = Ned_end
Nzl_end = Nzl_start + Nzl_rows
Spa_start = Nzl_end
Spa_end = Spa_start + Spa_rows
Swe_start = Spa_end
Swe_end = Swe_start + Swe_rows
Usa_start = Swe_end 
Usa_end = Usa_start + Usa_rows

#Use pandas concatenate to put all country data into one dataframe
CV_data = pd.concat((Bel_dat,Fra_dat,Ger_dat,Ita_dat,Ned_dat,Nzl_dat,Spa_dat,Swe_dat,Usa_dat),axis=0)

#Seperate dataframe into X and Y
X = CV_data.iloc[:,0:-1]
Y = CV_data.iloc[:,-1]

#Convert X and Y dataframes into numpy arrays
Xnp = X.to_numpy()
Ynp = Y.to_numpy()

#Create more features by multiplying features together - multiply Xnp by each individual Xnp Column (elementwise multiplication)
X0 = np.multiply(Xnp[0,:],Xnp)
X1 = np.multiply(Xnp[1,:],Xnp)
X2 = np.multiply(Xnp[2,:],Xnp)
X3 = np.multiply(Xnp[3,:],Xnp)
X4 = np.multiply(Xnp[4,:],Xnp)
X_mult = np.concatenate((Xnp,X0,X1,X2,X3,X4), axis=1)

#scale all X_mult columns between 0 and 1
X_sc = scaler.fit_transform(X_mult)

#Create X^2 and X^3 features
X2 = scaler.fit_transform(Xnp**2)
X3 = scaler.fit_transform(Xnp**3)

#Create Xtanh features from scaled X values
# (x*X_sc-y) x compresses shape of function, y offsets shape of function along x-axis
# +1 offsets shape of funtction along y axis to keep all y values between 0 and 2
# same term is subtracted with greater offset to create 'hump'
Xtanh= (np.tanh(4*X_sc-3.5)+1-(np.tanh(4*X_sc-4)+1))

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

#plt.subplot(321)
#plt.plot(Xnp[Bel_start:Bel_end,0],y_pred[Bel_start:Bel_end], label = "predicted")
#plt.plot(Xnp[Bel_start:Bel_end,0],Ynp[Bel_start:Bel_end], label = "actual")
#plt.xlabel('days after 5th case')
#plt.ylabel('deaths')
#plt.title('Belgium')

#plt.subplot(322)
#plt.plot(Xnp[Fra_start:Fra_end,0],y_pred[Fra_start:Fra_end], label = "predicted")
#plt.plot(Xnp[Fra_start:Fra_end,0],Ynp[Fra_start:Fra_end], label = "actual")
#plt.xlabel('days after 5th case')
#plt.ylabel('deaths')
#plt.title('France')

#plt.subplot(323)
#plt.plot(Xnp[Ger_start:Ger_end,0],y_pred[Ger_start:Ger_end], label = "predicted")
#plt.plot(Xnp[Ger_start:Ger_end,0],Ynp[Ger_start:Ger_end], label = "actual")
#plt.xlabel('days after 5th case')
#plt.ylabel('deaths')
#plt.title('Germany')

plt.subplot(321)
plt.plot(Xnp[Ita_start:Ita_end,0],y_pred[Ita_start:Ita_end], label = "predicted")
plt.plot(Xnp[Ita_start:Ita_end,0],Ynp[Ita_start:Ita_end], label = "actual")
plt.xlim(0,60)
plt.ylim(0,40000)
plt.xlabel('days after 5th case')
plt.ylabel('deaths')
plt.title('Italy')

plt.subplot(322)
plt.plot(Xnp[Ned_start:Ned_end,0],y_pred[Ned_start:Ned_end], label = "predicted")
plt.plot(Xnp[Ned_start:Ned_end,0],Ynp[Ned_start:Ned_end], label = "actual")
plt.xlim(0,60)
plt.ylim(0,40000)
plt.xlabel('days after 5th case')
plt.ylabel('deaths')
plt.title('Netherlands')

plt.subplot(323)
plt.plot(Xnp[Nzl_start:Nzl_end,0],y_pred[Nzl_start:Nzl_end], label = "predicted")
plt.plot(Xnp[Nzl_start:Nzl_end,0],Ynp[Nzl_start:Nzl_end], label = "actual")
plt.xlim(0,60)
plt.ylim(0,40000)
plt.xlabel('days after 5th case')
plt.ylabel('deaths')
plt.title('New Zealand')

plt.subplot(324)
plt.plot(Xnp[Spa_start:Spa_end,0],y_pred[Spa_start:Spa_end], label = "predicted")
plt.plot(Xnp[Spa_start:Spa_end,0],Ynp[Spa_start:Spa_end], label = "actual")
plt.xlim(0,60)
plt.ylim(0,40000)
plt.xlabel('days after 5th case')
plt.ylabel('deaths')
plt.title('Spain')

plt.subplot(325)
plt.plot(Xnp[Swe_start:Swe_end,0],y_pred[Swe_start:Swe_end], label = "predicted")
plt.plot(Xnp[Swe_start:Swe_end,0],Ynp[Swe_start:Swe_end], label = "actual")
plt.xlim(0,60)
plt.ylim(0,40000)
plt.xlabel('days after 5th case')
plt.ylabel('deaths')
plt.title('Sweden')

plt.subplot(326)
plt.plot(Xnp[Usa_start:Usa_end,0],y_pred[Usa_start:Usa_end], label = "predicted")
plt.plot(Xnp[Usa_start:Usa_end,0],Ynp[Usa_start:Usa_end], label = "actual")
plt.xlim(0,60)
plt.ylim(0,40000)
plt.xlabel('days after 5th case')
plt.ylabel('deaths')
plt.title('Usa')

plt.show()