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

input_file_3yr = "../3yr4mon_1day.csv"

print('input_file_3yr length', len(input_file_3yr))

forecastCandle = 0
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1-forecastCandle):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		# print('dataX', dataX)
		dataY.append(dataset[i + look_back + forecastCandle, 3])
		# print('dataY', dataY)
	return np.array(dataX), np.array(dataY)

# fix random seed for reproducibility
np.random.seed(5)

# load the dataset
df = read_csv(input_file_3yr, header=None, index_col=None, delimiter=',', usecols=[1,2,3,4])
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
dataset = scaler.fit_transform(dataset)

dataset=dataset.reshape(-1, 4)

print('dataset length', len(dataset))

look_back = 20
# split into train and test sets, 50% test data, 50% training data
#size of 1 year data
train_size = 365
dataset_len = len(dataset) 
print(len(dataset))
test_size = len(dataset) - train_size + look_back
train, test, train_volume_dataset, test_volume_dataset = dataset[0:train_size,:], dataset[train_size - look_back - (forecastCandle+1):train_size + (forecastCandle+1),:], df2_volume[0:train_size-1,:], df2_volume[train_size - look_back - (forecastCandle+1)-1:train_size + (forecastCandle+1)-1]

# reshape into X=t and Y=t+1, timestep 240
print('train ', train[0:2])
print('test ', test[0:2])
print('train_volume_dataset', train_volume_dataset[0:2])
#print(train[len(train)-20:])
#print(test[look_back+forecastCandle])
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

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


# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 4, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 4, testX.shape[1]))


# create and fit the LSTM network, optimizer=adam, 25 neurons, dropout 0.1
model = Sequential()
model.add(LSTM(25, input_shape=(4, look_back)))
model.add(Dropout(0.1))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(trainX, trainY, epochs=20, batch_size=60, verbose=1)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

print(len(trainPredict))
print(trainPredict[0])

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

#print('trainX[first]')
#print(trainX[0])
#print('trainX[last]')
#print(trainX[len(trainX)-1])
#print('testX[first]')
#print(testX[0])
#print('testX[last]')
#print(testX[len(testX) - 1])
print('train len:', len(trainY))
print(trainY[0])
print('test len:', len(testY))
print(testY)
print(len(testY))
print(len(testPredict))
#print(testX[len(testX)-1])
#print(scaler.inverse_transform([[0.04405421]]))
#print(scaler.inverse_transform([[0.044367921]]))



# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
#testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1-(forecastCandle*2), :] = testPredict

# plot baseline and predictions
#plt.plot(scaler.inverse_transform(dataset))
#plt.plot(trainPredictPlot)
#print('testPrices:')
arr2 = testYArr
print('arr2', arr2)

train_volume_dataset = train_volume_dataset[-1:]
print('train_volume_dataset', train_volume_dataset)
print('test_volume_dataset', test_volume_dataset[-1:])

#entry price
trainY = trainY.reshape(-1, 1)
trainY = trainY[-1:]

print('trainY2', trainY)
print('testPredictions:')
print(testPredict)
print(len(testPredict))

print('dataset length', len(dataset))
# print('dataset length', len(trainX))

# export prediction and actual prices
df = pd.DataFrame(data={"prediction": np.around(list(testPredict.reshape(-1)), decimals=2), "test_price": np.around(list(arr2.reshape(-1)), decimals=2), "volume": np.around(list(train_volume_dataset.reshape(-1)), decimals=2), "entry_test_price": np.around(list(trainY.reshape(-1)), decimals=2)})
file_name = "bit_1day_trade_examples_for_nn2.csv" 
df.to_csv(file_name, sep=';', index=None)
#df.to_json("testJson.json", orient = 'records')

# plot the actual price, prediction in test data=red line, actual price=blue line
#plt.plot(testPredictPlot)
#plt.show()
step = 1
for i in range(365+step, 1095, step):
	train_size = i
	dataset_len = len(dataset) 
	# print(len(dataset))
	test_size = len(dataset) - train_size + look_back
	# train, test, train_volume_dataset = dataset[train_size-look_back-(forecastCandle+1+step):train_size,:], dataset[train_size - look_back - (forecastCandle+1):train_size + (forecastCandle+1),:], df2_volume[train_size - look_back - (forecastCandle+1):train_size + (forecastCandle+1)]
	# Need to keep track of volume data in case we include it in price prediction as an input for future cases(added -1 in each index)
	train, test, train_volume_dataset, test_volume_dataset = dataset[train_size-look_back-(forecastCandle+1+step)-9:train_size,:], dataset[train_size - look_back - (forecastCandle+1):train_size + (forecastCandle+1),:], df2_volume[train_size-look_back-(forecastCandle+1+step)-1:train_size-1,:], df2_volume[train_size - look_back - (forecastCandle+1)-1:train_size + (forecastCandle+1)-1]

	# reshape into X=t and Y=t+1, timestep 240
	# print(len(train))
	# print(len(test))
	#print(train[len(train)-20:])
	#print(test[look_back+forecastCandle])
	trainX, trainY = create_dataset(train, look_back)
	testX, testY = create_dataset(test, look_back)

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

	# print(len(trainX))
	# print(len(testX))
	# print(len(testY))

	# reshape input to be [samples, time steps, features]
	trainX = np.reshape(trainX, (trainX.shape[0], 4, trainX.shape[1]))
	testX = np.reshape(testX, (testX.shape[0], 4, testX.shape[1]))

	# create and fit the LSTM network, optimizer=adam, 25 neurons, dropout 0.1
	#model = Sequential()
	#model.add(LSTM(25, input_shape=(1, look_back)))
	#model.add(Dropout(0.1))
	#model.add(Dense(1))
	#model.compile(loss='mse', optimizer='adam')
	model.fit(trainX, trainY, epochs=20, batch_size=60, verbose=1)

	# make predictions
	trainPredict = model.predict(trainX)
	testPredict = model.predict(testX)

	# invert predictions
	trainPredict = scaler.inverse_transform(trainPredict)
	trainY = scaler.inverse_transform([trainY])
	testPredict = scaler.inverse_transform(testPredict)
	testY = scaler.inverse_transform([testY])

	#print('trainX[first]')
	#print('trainX[first]')
	#print(trainX[0])
	#print('trainX[last]')
	#print(trainX[len(trainX)-1])
	#print('testX[first]')
	#print(testX[0])
	#print('testX[last]')
	#print(testX[len(testX) - 1])
	# print('train len:', len(trainY))
	# print(trainY[0])
	# print('test len:', len(testY))
	# print(testY[0])
	# print(len(testY))
	# print(len(testPredict))
	#print(testX[len(testX)-1])
	#print(scaler.inverse_transform([[0.04293486]]))
	#print(scaler.inverse_transform([[0.04352662]]))
	#print(scaler.inverse_transform([[0.04405421]]))
	#print(scaler.inverse_transform([[0.044367921]]))



	# calculate root mean squared error
	trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
	# print('Train Score: %.2f RMSE' % (trainScore))
	testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
	# print('Test Score: %.2f RMSE' % (testScore))

	# shift train predictions for plotting
	trainPredictPlot = np.empty_like(dataset)
	trainPredictPlot[:, :] = np.nan
	trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

	# shift test predictions for plotting
	testPredictPlot = np.empty_like(dataset)
	testPredictPlot[:, :] = np.nan
	
	arr2 = testYArr
	# print('arr2', arr2)
	# print('train_volume_dataset', train_volume_dataset)
	# print('test_volume_dataset', test_volume_dataset)

	train_volume_dataset = train_volume_dataset[-1:]
	# print('train_volume_dataset', train_volume_dataset)
	# print('test_volume_dataset', test_volume_dataset[-1:])
	# arr2 = []
	# for val in testPrices:
	#     arr2.append([val[3]])

	# arr2 = np.array(arr2)
	# arr2.reshape(-1,1)

	# print(arr2)
	#entry price
	trainY = trainY.reshape(-1, 1)
	trainY = trainY[-1:]

	train_volume_dataset = train_volume_dataset[-1:]

	# export prediction and actual prices
	df = pd.DataFrame(data={"prediction": np.around(list(testPredict.reshape(-1)), decimals=2), "test_price": np.around(list(arr2.reshape(-1)), decimals=2), "volume": np.around(list(train_volume_dataset.reshape(-1)), decimals=2), "entry_test_price": np.around(list(trainY.reshape(-1)), decimals=2)})
	#file_name = "lstm_result_5min_x_is_10_retraining2"+ str(train_size)+ ".csv" 
	df.to_csv(file_name, sep=';', mode = 'a', index=None, header=None)

	# plot the actual price, prediction in test data=red line, actual price=blue line
	#plt.plot(testPredictPlot)
	#plt.show()

