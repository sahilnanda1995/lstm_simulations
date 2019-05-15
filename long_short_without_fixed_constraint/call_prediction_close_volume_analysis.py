import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers.core import Dense, Activation, Dropout
import time #helper libraries

# file is downloaded from finance.yahoo.com, 1.1.1997-1.1.2017
# training data = 1.1.1997 - 1.1.2007
# test data = 1.1.2007 - 1.1.2017

input_file_3yr = "../3yr4mon1hr_bit.csv"

print('input_file_3yr length', len(input_file_3yr))

forecastCandle = 0
# convert an array of values into a dataset matrix
def create_dataset(dataset, volume_dataset, look_back=1):
	dataX, dataY = [], []
	dataX2, dataY2 = [], []
	for i in range(len(dataset)-look_back-1-forecastCandle):
		b = []
		a = dataset[i:(i+look_back)]
		vol = volume_dataset[i:(i+look_back)]
		a = scaler.fit_transform(a)
		vol = scaler.fit_transform(vol)
		for x in range(len(a)):
			b.append(a[x])
			b.append(vol[x])
		b = np.array(b)
		b = b.reshape(-1,2)
		# print('b', b)
		a = b
		dataX.append(a)
		# print('a', a[-10:])
		# print('vol', vol[-10:])
		# print('index', np.argmax(a[-10:], axis = 0)[0])
		# print('index', np.argmin(a[-10:], axis = 0)[0])
		# print('dataX', dataX)
		for j in range(1,11):
			if j!=10:
				dataX2.append(a[-20+j:-10+j])
			else:
				dataX2.append(a[-20+j:])
			if j == (np.argmin(a[-10:], axis = 0)[0]+1):
				dataY2.append([1, 0, 0]) # long
			elif j == (np.argmax(a[-10:], axis = 0)[0]+1):
				dataY2.append([0, 1, 0]) # short
			else:
				dataY2.append([0, 0, 1]) # do nothing
			# dataX2.append(a[-20+j:-10+j])
		# print('dataX2', dataX2)
		# print('dataY2', dataY2)
		# dataY.append(dataset[i + look_back + forecastCandle, 3])
		# print('dataY', dataY)
	return np.array(dataX2), np.array(dataY2)


def create_dataset2(dataset, volume_dataset, look_back=1):
	dataX, dataY = [], []
	dataX2, dataY2 = [], []
	for i in range(len(dataset)-look_back-1-forecastCandle):
		b = []
		a = dataset[i:(i+look_back)]
		vol = volume_dataset[i:(i+look_back)]
		a = scaler.fit_transform(a)
		vol = scaler_vol.fit_transform(vol)
		for x in range(len(a)):
			b.append(a[x])
			b.append(vol[x])
		b = np.array(b)
		b = b.reshape(-1,2)
		# print('b', b)
		a = b
		dataX.append(a)
		# print('a', a[-10:])
		# print('vol', vol[-10:])
		# print('index', np.argmax(a[-10:], axis = 0)[0])
		# print('index', np.argmin(a[-10:], axis = 0)[0])
		# print('dataX', dataX)
		# for j in range(1,11):
		# 	if j!=10:
		# 		dataX2.append(a[-20+j:-10+j])
		# 	else:
		# 		dataX2.append(a[-20+j:])
		# 	if j == (np.argmin(a[-10:], axis = 0)[0]+1):
		# 		dataY2.append([1, 0, 0]) # long
		# 	elif j == (np.argmax(a[-10:], axis = 0)[0]+1):
		# 		dataY2.append([0, 1, 0]) # short
		# 	else:
		# 		dataY2.append([0, 0, 1]) # do nothing

		dataX2.append(a[-10:])
		dataY2.append(scaler.inverse_transform([[a[len(a)-1][0]]]))
		# print('dataX2', dataX2)
		# print('dataY2', dataY2)
		# dataY.append(dataset[i + look_back + forecastCandle, 3])
		# print('dataY', dataY)
	return np.array(dataX2), np.array(dataY2)



# fix random seed for reproducibility
np.random.seed(5)

# load the dataset
df = read_csv(input_file_3yr, header=None, index_col=None, delimiter=',', usecols=[4])
# df2 = read_csv(input_file_3yr, header=None, index_col=None, delimiter=',', usecols=[0,1,2,3])
df2_volume = read_csv(input_file_3yr, header=None, index_col=None, delimiter=',', usecols=[5])

print('volumelength', len(df2_volume))
# print('df length', len(df))
# print('df all', df.values)
# print('df2 all', df2.values)
# print('df2_volume all', df2_volume.values)
# take close price column[5]
all_y = df.values
print(all_y[0:10])

dataset=all_y.reshape(-1, 1)

volume_all_y = df2_volume.values

df2_volume = volume_all_y.reshape(-1, 1)

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
scaler_vol = MinMaxScaler(feature_range=(0, 1))
# dataset = scaler.fit_transform(dataset)

dataset=dataset.reshape(-1, 1)

print('dataset length', len(dataset))

finding_min_max_array_len = 10
look_back = 20
parent_array_len = finding_min_max_array_len + look_back

# split into train and test sets, 50% test data, 50% training data
#size of 1 year data
train_size = 26300
dataset_len = len(dataset) 
print(len(dataset))
test_size = len(dataset) - train_size + look_back
train, test, train_volume_dataset, test_volume_dataset = dataset[0:train_size,:], dataset[train_size - look_back - (forecastCandle+1):,:], df2_volume[0:train_size,:], df2_volume[train_size - look_back - (forecastCandle+1):,:]

# reshape into X=t and Y=t+1, timestep 240
print('train ', train[0:2])
print('test ', test[0:2])
print('train_volume_dataset', train_volume_dataset[0:2])
#print(train[len(train)-20:])
#print(test[look_back+forecastCandle])
trainX, trainY = create_dataset(train, train_volume_dataset, look_back)
testX, testY = create_dataset2(test, test_volume_dataset, look_back)

print(len(testX))

print(trainX.shape)
print(trainY.shape)

# trainXArr = []
# for val in trainX[len(trainX)-1]:
# 	trainXArr.append(val[3])

# trainXArr = np.array(trainXArr)
# trainXArr = trainXArr[-10:]
# trainXArr = trainXArr.reshape(-1,1)
# print(trainXArr)
# trainXArr = scaler.inverse_transform(trainXArr)

# print('trainXArr', trainXArr)

# trainYArr = trainY
# trainYArr = np.array(trainYArr)
# trainYArr = trainYArr.reshape(-1, 1)
# trainYArr = scaler.inverse_transform(trainYArr)
# print('trainYArr', trainYArr)

# testXArr = []
# for val in testX[len(testX)-1]:
# 	testXArr.append(val[3])


# testXArr = np.array(testXArr)
# testXArr = testXArr[-10:]
# testXArr = testXArr.reshape(-1,1)
# print(testXArr)
# testXArr = scaler.inverse_transform(testXArr)

# print('testXArr', testXArr)

# testYArr = testY
# testYArr = np.array(testYArr)
# testYArr = testYArr.reshape(-1, 1)
# testYArr = scaler.inverse_transform(testYArr)
# print('testYArr', testYArr)


print(trainX.shape)
print(trainY.shape)

print('trainY_no_re', trainY)

# trainY = trainY.reshape(-1,1,3)

print('trainY_re', trainY)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 2, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 2, testX.shape[1]))


print('trainX', trainX[-2:])
print(trainX.shape)
print(len(trainX))

# create and fit the LSTM network, optimizer=adam, 25 neurons, dropout 0.1
model = Sequential()
model.add(LSTM(25, input_shape=(2, 10)))
model.add(Dense(25, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(trainX, trainY, epochs=20, batch_size=60, verbose=1)


# plt.plot(history.history['mean_absolute_error'])
# plt.plot(history.history['val_mean_absolute_error'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

long_Arr = []
short_Arr = []
do_nothing_Arr = []

print(len(testPredict))
for i in range(len(testPredict)):
	print(testPredict[i], testY[i])
	long_Arr.append([testPredict[i][0]])
	short_Arr.append([testPredict[i][1]])
	do_nothing_Arr.append([testPredict[i][2]])
	print(long_Arr[i], short_Arr[i], do_nothing_Arr[i])


long_Arr = np.array(long_Arr)
short_Arr = np.array(short_Arr)
do_nothing_Arr = np.array(do_nothing_Arr)


df = pd.DataFrame(data={"price": np.around(list(testY.reshape(-1)), decimals=2), "long_prob": np.around(list(long_Arr.reshape(-1)), decimals=8), "short_prob": np.around(list(short_Arr.reshape(-1)), decimals=8), "do_nothing_prob": np.around(list(do_nothing_Arr.reshape(-1)), decimals=8)})
file_name = "long_short_prob.csv" 
df.to_csv(file_name, sep=';', index=None)
