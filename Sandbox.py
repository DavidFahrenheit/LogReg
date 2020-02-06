
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

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 1)


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

#print(Y_train.head)