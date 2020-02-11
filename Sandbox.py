
import numpy as np
import pandas as pd
import sklearn



#with open('DxG.txt', newline = '') as DxG:
  #xG_Data= csv.reader(DxG, delimiter='\t')

xG_Data = pd.read_csv('DxG.txt', sep ="\s+")
print(xG_Data.shape)

X = xG_Data.iloc[:,0:-1]

Y = xG_Data.iloc[:,-1]

#xG_Data['target'].value_counts()

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression(max_iter = 1000)

logmodel.fit(X_train, Y_train)

y_pred = logmodel.predict(X_test)	

print('Accuracy: %d', (logmodel.score(X_test, Y_test)))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_test,y_pred)
print(confusion_matrix)
#print(len(xG_Data))
#print(type(xG_Data))
#print(xG_Data.head)

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

Mtheta = logmodel.coef_ 
Mintercept = logmodel.intercept_
print(Mtheta)
print(Mintercept)

X_trainnp = X_train.to_numpy()


print(type(X_trainnp))
print(X_trainnp[6:7,:].shape)

Mtheta_T = Mtheta.transpose()

xG_test = sigmoid(Mintercept + np.matmul(X_trainnp[16], Mtheta_T))
print(xG_test)
