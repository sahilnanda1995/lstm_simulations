
# coding: utf-8

# In[1]:


from keras.layers.core import Dense, Activation, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import matplotlib.pyplot as plt
from pandas import read_csv
import pandas as pd
import numpy as np
import math
import time

np.random.seed(5)


# In[2]:


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1-forecastCandle):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back + forecastCandle, 3])
    return np.array(dataX), np.array(dataY)

def model1(look_back = 20):
    model = Sequential()
    model.add(LSTM(25, input_shape=(4, look_back)))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam',metrics=['mae'])
    return model

def model2(look_back = 20):
    model = Sequential()
    model.add(LSTM(input_shape=(4, look_back), units = 30, 
                   return_sequences = True))
    model.add(Dropout(0.5))
    model.add(LSTM(256))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation("linear"))
    model.compile(loss='mse', optimizer='adam',metrics=['mae'])
    return model

def model3(look_back = 20):
    model = Sequential()
    model.add(LSTM(30, input_shape=(4, look_back), return_sequences=True))
    model.add(LSTM(units=30, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(units=30))
    model.add(Dense(units=1))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam',metrics=['mae'])
    return model


# In[3]:


input_file_3yr = "../3yr4mon5min_bit.csv"
forecastCandle = 9
df = read_csv(input_file_3yr, header=None, index_col=None, delimiter=',', usecols=[1,2,3,4])
df2_volume = read_csv(input_file_3yr, header=None, index_col=None, delimiter=',', usecols=[5])
all_y = df.values
dataset=all_y.reshape(-1, 1)
volume_all_y = df2_volume.values
df2_volume = volume_all_y.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
dataset=dataset.reshape(-1, 4)
look_back = 20
train_size = 352420
dataset_len = len(dataset) 
test_size = len(dataset) - train_size + look_back
train, test, volume_dataset = dataset[0:train_size,:], dataset[train_size - look_back - (forecastCandle+1):train_size + (forecastCandle+1),:], df2_volume[train_size - look_back - (forecastCandle+1):train_size + (forecastCandle+1)]


# In[4]:


trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

trainXArr = []
for val in trainX[len(trainX)-1]:
    trainXArr.append(val[3])

trainXArr = np.array(trainXArr)
trainXArr = trainXArr[-10:]
trainXArr = trainXArr.reshape(-1,1)
trainXArr = scaler.inverse_transform(trainXArr)
print('trainXArr', trainXArr)

trainYArr = trainY
trainYArr = np.array(trainYArr)
trainYArr = trainYArr.reshape(-1, 1)
trainYArr = scaler.inverse_transform(trainYArr)
print('trainYArr', trainYArr)

testXArr = []
for val in testX[len(testX)-1]:
    testXArr.append(val[3])
testXArr = np.array(testXArr)
testXArr = testXArr[-10:]
testXArr = testXArr.reshape(-1,1)
testXArr = scaler.inverse_transform(testXArr)
print('testXArr', testXArr)

testYArr = testY
testYArr = np.array(testYArr)
testYArr = testYArr.reshape(-1, 1)
testYArr = scaler.inverse_transform(testYArr)
print('testYArr', testYArr)

trainX = np.reshape(trainX, (trainX.shape[0], 4, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 4, testX.shape[1]))


# In[9]:


model = model1(look_back)
history = model.fit(trainX, trainY, epochs=10, batch_size=60, verbose=1,validation_split=0.3)


# In[11]:



plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

