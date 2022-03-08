#導入標準庫
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#導入資料
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#名義變量進行編碼處理
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
ct=ColumnTransformer([('Country',OneHotEncoder(),[1])],remainder='passthrough')
X = ct.fit_transform(X)
X = X[:, 1:] #避免虛擬變量陷阱，原本國家是三個變量，後來改為兩個

#將資料分成訓練集以及測試集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#特徵縮放
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#導入Keras的標準庫
import keras
from keras.models import Sequential
from keras.layers import Dense

#創建ANN
classifier = Sequential()

#增加輸入層以及第一隱藏層
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

#加入第二層隱藏層
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

#加入輸出層
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

#編譯ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#將訓練集投入模型進行訓練
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

#預測測試結果
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#創建混淆矩陣並印出
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix: \n', cm)