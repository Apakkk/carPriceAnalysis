import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error,mean_absolute_error

dataFrame = pd.read_excel("merc.xlsx")
#print(dataFrame.head())
#print(dataFrame.isnull().sum())
plt.figure(figsize=(7,5))
#print(sbn.displot(dataFrame["price"]))

sbn.countplot(dataFrame["year"])

#dataFrame.corr()
#dataFrame.corr()["price"].sort_values()
#sbn.scatterplot(x = "mileage",y="price",data = dataFrame)

#print(dataFrame.sort_values("price",ascending=False).head(20))

len(dataFrame)*0.01 #tane en yüksek fiyatlı araba silinecek

ninetyNinePercentDf = dataFrame.sort_values("price",ascending=False).iloc[131:] #en yüksekleri görmezden geldik

#print(ninetyNinePercentDf.describe())

#plt.figure(figsize=(7,5)) #grafik çizimi için en boy

#sbn.displot(ninetyNinePercentDf["price"]) #grafik

#ninetyNinePercentDf.groupby("year").mean()["price"]

#dataFrame[dataFrame.year != 1970].groupby("year").mean()["price"]

dataFrame = ninetyNinePercentDf

dataFrame = dataFrame[dataFrame.year != 1970]

#dataFrame.groupby("year").mean()["price"]

dataFrame = dataFrame.drop("transmission",axis=1)

y = dataFrame["price"].values
x = dataFrame.drop("price",axis=1).values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=10)

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test)

#print(x_train.shape)

model = Sequential()

model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))

model.add(Dense(1))

model.compile(optimizer = "adam",loss = "mse")

model.fit(x = x_train,y = y_train,validation_data = (x_test,y_test),batch_size = 250,epochs = 300)

lossData = pd.DataFrame(model.history.history)

#print(lossData.head())

#lossData.plot()

guessArray = model.predict(x_test)

mean_absolute_error(y_test,guessArray)

#dataFrame.describe()

graph1 = plt.scatter(y_test,guessArray)

graph2 = plt.plot(y_test,y_test,"g-*")

dataFrame.iloc[2]

newCarSeries = dataFrame.drop("price",axis = 1).iloc[2]

newCarSeries = scaler.transform(newCarSeries.values.reshape(-1,5))

model.predict(newCarSeries)



