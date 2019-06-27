import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Bidirectional
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers.core import Dense, Activation, Dropout
import time
# import nn2_for_1dayCandle #helper libraries


input_file_3yr = "../3yr4mon1hr_bit.csv"
print('input_file_3yr length', len(input_file_3yr))

forecastCandle = 0
# convert an array of values into a dataset matrix
def create_dataset(dataset, vol_dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1-forecastCandle):
		c = []
		a = dataset[i:(i+look_back)]
		# print('a', a)
		a = a.reshape(-1, 1)
		a = scaler.fit_transform(a)
		a = a.reshape(-1, 4)
		b = vol_dataset[i:(i+look_back)]
		# print('a', a)
		# print('b', b)
		b = scaler_vol.fit_transform(b)
		# print('b', b)
		# a = a.reshape(-1, 1)
		for j in range(len(b)):
			for k in range(len(a[j])):
				c.append(a[j][k])
			c.append(b[j])
		c = np.array(c)
		c = c.reshape(-1, 1)
		c = c.reshape(-1, 5)
		dataX.append(c)
		# print('dataX', dataX)
		y = dataset[i + look_back + forecastCandle, 1:3]
		# print('y', y)
		y = scaler.transform([y])
		# print('y', y[0])
		dataY.append(y[0])
		# print('dataY', dataY)
	return np.array(dataX), np.array(dataY)
# fix random seed for reproducibility
np.random.seed(5)

# load the dataset
df = read_csv(input_file_3yr, header=None, index_col=None, delimiter=',', usecols=[1,2,3,4])

df2_volume = read_csv(input_file_3yr, header=None, index_col=None, delimiter=',', usecols=[5])

df3_timeStamp = read_csv(input_file_3yr, header=None, index_col=None, delimiter=',', usecols=[0])

print('volumelength', len(df2_volume))

all_y = df.values
print(all_y[0:10])

dataset=all_y.reshape(-1, 1)

volume_all_y = df2_volume.values

df2_volume = volume_all_y.reshape(-1, 1)

df3_timeStamp = df3_timeStamp.values

df3_timeStamp = df3_timeStamp.reshape(-1, 1)

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
# dataset = scaler.fit_transform(dataset)

dataset=dataset.reshape(-1, 4)

print('dataset length', len(dataset))


scaler_vol = MinMaxScaler(feature_range=(0, 1))
# df2_volume = scaler_vol.fit_transform(df2_volume)

look_back = 20
# split into train and test sets, 50% test data, 50% training data
#size of 1 year data
train_size = 30004
dataset_len = len(dataset) 
print(len(dataset))
test_size = len(dataset) - train_size + look_back
train, test, train_volume_dataset, test_volume_dataset = dataset[0:train_size,:], dataset[train_size - look_back - (forecastCandle+1):train_size + (forecastCandle+1),:], df2_volume[0:train_size,:], df2_volume[train_size - look_back - (forecastCandle+1):train_size + (forecastCandle+1),:]

train_timeStamp, test_timeStamp = df3_timeStamp[0:train_size-1,:], df3_timeStamp[train_size - look_back - (forecastCandle+1)-1:train_size + (forecastCandle+1)-1] 

# reshape into X=t and Y=t+1, timestep 240
print('train ', train[0:2])
print('test ', test[0:2])
print('train_volume_dataset', train_volume_dataset[0:2])
#print(train[len(train)-20:])
#print(test[look_back+forecastCandle])
trainX, trainY = create_dataset(train, train_volume_dataset, look_back)
testX, testY = create_dataset(test, test_volume_dataset, look_back)

trainXArr = []
for val in trainX[len(trainX)-1]:
	trainXArr.append(val[3])

trainXArr = np.array(trainXArr)
trainXArr = trainXArr[-10:]
trainXArr = trainXArr.reshape(-1,1)
print(trainXArr)
trainXArr = scaler.inverse_transform(trainXArr)

print('trainXArr', trainXArr)

trainYArr = trainY
trainYArr = np.array(trainYArr)
trainYArr = trainYArr.reshape(-1, 1)
trainYArr = scaler.inverse_transform(trainYArr)
trainYArr = trainYArr.reshape(-1, 2)
print('trainYArr', trainYArr)

testXArr = []
for val in testX[len(testX)-1]:
	testXArr.append(val[3])


testXArr = np.array(testXArr)
testXArr = testXArr[-10:]
testXArr = testXArr.reshape(-1,1)
print(testXArr)
testXArr = scaler.inverse_transform(testXArr)

print('testXArr', testXArr)

testYArr = testY
testYArr = np.array(testYArr)
testYArr = testYArr.reshape(-1, 1)
testYArr = scaler.inverse_transform(testYArr)
print('testYArr', testYArr)

test = scaler.inverse_transform(test)
test= test[-2:-1]
test = test.reshape(-1, 1)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 5, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 5, testX.shape[1]))


# create and fit the LSTM network, optimizer=adam, 25 neurons, dropout 0.1

window_size, input_shape, dropout_value, activation_function, loss_function, optimizer = look_back, 5, 0.2, 'linear', 'mse', 'adam'

# create and fit the LSTM network, optimizer=adam, 25 neurons, dropout 0.1
model = Sequential()
model.add(Bidirectional(LSTM(window_size, return_sequences=True), input_shape=(input_shape, window_size),))
model.add(Dropout(dropout_value))
model.add(Bidirectional(LSTM((window_size*2), return_sequences=True)))
model.add(Dropout(dropout_value))
model.add(Bidirectional(LSTM(window_size, return_sequences=False)))
model.add(Dense(units=2))
model.add(Activation(activation_function))
model.compile(loss=loss_function, optimizer=optimizer)
model.fit(trainX, trainY, batch_size= 60, nb_epoch=30)


# model = Sequential()
# model.add(LSTM(25, input_shape=(5, look_back)))
# model.add(Dropout(0.1))
# model.add(Dense(2))
# model.compile(loss='mse', optimizer='adam')
# model.fit(trainX, trainY, epochs=20, batch_size=60, verbose=1)



# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

print(len(trainPredict))
print(trainPredict[0])

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = trainY.reshape(-1, 1)
trainY = scaler.inverse_transform(trainY)
trainY = trainY.reshape(-1, 2)
testPredict = scaler.inverse_transform(testPredict)
testY = testY.reshape(-1, 1)
testY = scaler.inverse_transform(testY)
testY = testY.reshape(-1, 2)

print('train len:', len(trainY))
print(trainY[0])
print('test len:', len(testY))
print(testY)
print(len(testY))
print(len(testPredict))


arr2 = testYArr
arr2 = arr2.reshape(-1, 1)
arr2 = arr2[-1:]
print('arr2', arr2)


print('train_timeStamp', train_timeStamp[-1:])

train_volume_dataset = train_volume_dataset[-1:]
print('train_volume_dataset', train_volume_dataset)
print('test_volume_dataset', test_volume_dataset[-1:])

#entry price
trainY = trainY.reshape(-1, 1)
trainY = trainY[-1:]

print('trainY2', trainY)
print('testPredictions:')
print(testPredict)
testPredict = testPredict.reshape(-1, 1)
print(len(testPredict))

print('dataset length', len(dataset))


df = pd.DataFrame(data={"timeStamp": np.around(list(train_timeStamp[-1].reshape(-1)), decimals=2), "high_prediction": np.around(list(testPredict[0:1].reshape(-1)), decimals=2), "low_prediction": np.around(list(testPredict[1:2].reshape(-1)), decimals=2), "test_price": np.around(list(arr2.reshape(-1)), decimals=2), "volume": np.around(list(train_volume_dataset.reshape(-1)), decimals=2), "entry_test_price": np.around(list(testXArr[len(testXArr)-1].reshape(-1)), decimals=2), "exit_high_price": np.around(list(testYArr[0].reshape(-1)), decimals=2), "exit_low_price": np.around(list(testYArr[1].reshape(-1)), decimals=2)})
file_name = "1hr_accuracy_model57.csv" 
df.to_csv(file_name, sep=';', index=None)

step = 1

for i in range(30004+step, len(dataset)-1, step):
	train_size = i
	dataset_len = len(dataset) 
	# print(len(dataset))
	test_size = len(dataset) - train_size + look_back
	# Need to keep track of volume data in case we include it in price prediction as an input for future cases(added -1 in each index)
	train, test, train_volume_dataset, test_volume_dataset = dataset[train_size-look_back-(forecastCandle+1+step)-9:train_size,:], dataset[train_size - look_back - (forecastCandle+1):train_size + (forecastCandle+1),:], df2_volume[train_size-look_back-(forecastCandle+1+step)-9:train_size,:], df2_volume[train_size - look_back - (forecastCandle+1):train_size + (forecastCandle+1),:]

	train_timeStamp, test_timeStamp = df3_timeStamp[train_size-look_back-(forecastCandle+1+step)-1:train_size-1,:], df3_timeStamp[train_size - look_back - (forecastCandle+1)-1:train_size + (forecastCandle+1)-1] 
	# reshape into X=t and Y=t+1, timestep 240
	# print(len(train))
	# print(len(test))
	#print(train[len(train)-20:])
	#print(test[look_back+forecastCandle])
	trainX, trainY = create_dataset(train, train_volume_dataset, look_back)
	testX, testY = create_dataset(test, test_volume_dataset, look_back)

	trainXArr = []
	for val in trainX[len(trainX)-1]:
		trainXArr.append(val[3])

	trainXArr = np.array(trainXArr)
	trainXArr = trainXArr[-10:]
	trainXArr = trainXArr.reshape(-1,1)
	# print(trainXArr)
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
	print(testXArr)
	testXArr = scaler.inverse_transform(testXArr)
	print('testXArr', testXArr)

	testYArr = testY
	testYArr = np.array(testYArr)
	testYArr = testYArr.reshape(-1, 1)
	testYArr = scaler.inverse_transform(testYArr)
	print('testYArr', testYArr)

	test = scaler.inverse_transform(test)
	test = test[-2:-1]
	test = test.reshape(-1, 1) 
	# print(len(trainX))
	# print(len(testX))
	# print(len(testY))

	# reshape input to be [samples, time steps, features]
	trainX = np.reshape(trainX, (trainX.shape[0], 5, trainX.shape[1]))
	testX = np.reshape(testX, (testX.shape[0], 5, testX.shape[1]))

	# create and fit the LSTM network, optimizer=adam, 25 neurons, dropout 0.1
	#model = Sequential()
	#model.add(LSTM(25, input_shape=(1, look_back)))
	#model.add(Dropout(0.1))
	#model.add(Dense(1))
	#model.compile(loss='mse', optimizer='adam')
	model.fit(trainX, trainY, batch_size= 60, nb_epoch=30)

	# make predictions
	trainPredict = model.predict(trainX)
	testPredict = model.predict(testX)

	# invert predictions
	trainPredict = scaler.inverse_transform(trainPredict)
	trainY = trainY.reshape(-1, 1)
	trainY = scaler.inverse_transform(trainY)
	trainY = trainY.reshape(-1, 2)
	testPredict = scaler.inverse_transform(testPredict)
	testY = testY.reshape(-1, 1)
	testY = scaler.inverse_transform(testY)
	testY = testY.reshape(-1, 2)
	
	arr2 = testYArr
	arr2 = arr2.reshape(-1, 1)
	arr2 = arr2[-1:]
	print('arr2', arr2)

	train_volume_dataset = train_volume_dataset[-1:]
	
	print('testPredictions:')
	print(testPredict)
	testPredict = testPredict.reshape(-1, 1)
	print(testPredict)
	trainY = trainY.reshape(-1, 1)
	trainY = trainY[-1:]

	train_volume_dataset = train_volume_dataset[-1:]
	print('train_timeStamp', train_timeStamp[-1:])
	df = pd.DataFrame(data={"timeStamp": np.around(list(train_timeStamp[-1].reshape(-1)), decimals=2), "high_prediction": np.around(list(testPredict[0:1].reshape(-1)), decimals=2), "low_prediction": np.around(list(testPredict[1:2].reshape(-1)), decimals=2), "test_price": np.around(list(arr2.reshape(-1)), decimals=2), "volume": np.around(list(train_volume_dataset.reshape(-1)), decimals=2), "entry_test_price": np.around(list(testXArr[len(testXArr)-1].reshape(-1)), decimals=2), "exit_high_price": np.around(list(testYArr[0].reshape(-1)), decimals=2), "exit_low_price": np.around(list(testYArr[1].reshape(-1)), decimals=2)})
	df.to_csv(file_name, sep=';', mode = 'a', index=None, header=None)

	
