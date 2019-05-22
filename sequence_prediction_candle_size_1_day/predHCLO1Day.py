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
import time
# import nn2_for_1hr
# import lstm_up_down_step_01 #helper libraries


input_file_3yr = "../bit_2013to2019_1dayCandle.csv"

print('input_file_3yr length', len(input_file_3yr))

forecastCandle = 0
sequenceNumber = 10
# convert an array of values into a dataset matrix
def create_dataset(dataset, vol_dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-10-forecastCandle):
		c = []
		a = dataset[i:(i+look_back)]
		# print('a', a[-10:])
		b = vol_dataset[i:(i+look_back)]
		# print('b', b[-10:])
		# a = a.reshape(-1, 1)
		for j in range(len(b)):
			for k in range(len(a[j])):
				c.append(a[j][k])
			c.append(b[j])
		c = np.array(c)
		c = c.reshape(-1, 1)
		c = c.reshape(-1, 5)
		# print('c', c)
		dataX.append(c)
		# print('dataX', dataX)
		# print('b', dataset[i + look_back + forecastCandle, 3])

		dataY.append(dataset[i + look_back + forecastCandle: i + look_back + forecastCandle + sequenceNumber, 3])
		# print('dataX', dataX)
		# print('dataY', dataY)
	return np.array(dataX), np.array(dataY)

# fix random seed for reproducibility
np.random.seed(5)

# load the dataset
df = read_csv(input_file_3yr, header=None, index_col=None, delimiter=',', usecols=[1,2,3,4])
# df2 = read_csv(input_file_3yr, header=None, index_col=None, delimiter=',', usecols=[0,1,2,3])
df2_volume = read_csv(input_file_3yr, header=None, index_col=None, delimiter=',', usecols=[5])

df3_timeStamp = read_csv(input_file_3yr, header=None, index_col=None, delimiter=',', usecols=[0])

df3_timeStamp = df3_timeStamp.values

df3_timeStamp = df3_timeStamp.reshape(-1, 1)


print('volumelength', len(df2_volume))

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
print('dataset', dataset[230:300])


scaler_vol = MinMaxScaler(feature_range=(0, 1))
df2_volume = scaler_vol.fit_transform(df2_volume)



look_back = 20
# split into train and test sets, 50% test data, 50% training data
#size of 1 year data
train_size = 1733
dataset_len = len(dataset) 
print(len(dataset))
test_size = len(dataset) - train_size + look_back
train, test, train_volume_dataset, test_volume_dataset = dataset[0:train_size,:], dataset[train_size - look_back - (forecastCandle+1):train_size + (forecastCandle+1) + (sequenceNumber - 1),:], df2_volume[0:train_size,:], df2_volume[train_size - look_back - (forecastCandle+1):train_size + (forecastCandle+1) + (sequenceNumber - 1),:]
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
print('trainY before reshape', trainY[0:10])
print('trainY before reshape', trainY[-10:])
trainYArr = np.array(trainYArr)
trainYArr = trainYArr.reshape(-1, 10)
print('trainYArrAfteReshapestart', trainYArr[0:10])
print('trainYArrAfteReshapeEnd', trainYArr[-10:])
trainYArr = scaler.inverse_transform(trainYArr)
print('trainYArr', trainYArr[-10:])


print(testX)
print(len(testX))
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
testYArr = testYArr.reshape(-1, 10)
testYArr = scaler.inverse_transform(testYArr)
print('testYArr', testYArr)


# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 5, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 5, testX.shape[1]))


# create and fit the LSTM network, optimizer=adam, 25 neurons, dropout 0.1
model = Sequential()
model.add(LSTM(25, input_shape=(5, look_back)))
model.add(Dropout(0.1))
model.add(Dense(10))
model.compile(loss='mse', optimizer='adam')
model.fit(trainX, trainY, epochs=64, batch_size=60, verbose=1)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

print(len(trainPredict))
print(trainPredict[0])

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = trainY.reshape(-1, 10)
trainY = scaler.inverse_transform(trainY)
print('trainY', trainY[-1:])
testPredict = scaler.inverse_transform(testPredict)
print('testPredict', testPredict)
trainY = trainY.reshape(-1, 10)
testY = scaler.inverse_transform(testY)
print(testY)
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
# trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
# print('Train Score: %.2f RMSE' % (trainScore))
# testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
# print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
# trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

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
print('train_timeStamp', train_timeStamp[-1:])

train_volume_dataset = train_volume_dataset[-1:]
print('train_volume_dataset', train_volume_dataset)
print('test_volume_dataset', test_volume_dataset[-1:])
# print('rsi', test_rsi_dataset[-30:])

#entry price
# trainY = trainY.reshape(-1, 1)
# trainY = trainY[-1:]

print('trainY2', trainY)
print('testPredictions:')
print(testPredict)
print(len(testPredict))

print('dataset length', len(dataset))


# callTakingProb = nn2_for_1hr.predict_value(trainY, testPredict, train_volume_dataset)
# upDownPred = lstm_up_down_step_01.predict_upDown(test_rsi_dataset)

# if testPredict[0][0] > trainY[0][0]:
#     actionTaken = 1
# else:
#     actionTaken = 0
# print('nn2_appending_inputs', trainY[0][0], testPredict[0][0], actionTaken, arr2[0][0], train_volume_dataset[0][0])
# nn2_for_1hr.appendLatestTradeExample(trainY[0][0], testPredict[0][0], actionTaken, arr2[0][0], train_volume_dataset[0][0])

# export prediction and actual prices

print(len(testPredict))
print(testPredict)
testPredict = testPredict.reshape(-1, 1)
testPredict1 = testPredict[0:1]
print(testPredict1)
testPredict10 = testPredict[-1:]
print(testPredict10)
print(len(arr2))
print(arr2)
arr2 = arr2.reshape(-1, 1)
arr2_1 = arr2[0:1]
print('arr2_1', arr2_1)
arr2_10 = arr2[-1:]
print('arr2_10', arr2_10)
print(len(testXArr))
print(testXArr)

testXArr = testXArr[-1:]
print(len(testXArr))
print(testXArr)

df = pd.DataFrame(data={"timeStamp": np.around(list(train_timeStamp[-1].reshape(-1)), decimals=2), "prediction01": np.around(list(testPredict[0:1].reshape(-1)), decimals=2), "prediction02": np.around(list(testPredict[1:2].reshape(-1)), decimals=2), "prediction03": np.around(list(testPredict[2:3].reshape(-1)), decimals=2), "prediction04": np.around(list(testPredict[3:4].reshape(-1)), decimals=2), "prediction05": np.around(list(testPredict[4:5].reshape(-1)), decimals=2), "prediction06": np.around(list(testPredict[5:6].reshape(-1)), decimals=2), "prediction07": np.around(list(testPredict[6:7].reshape(-1)), decimals=2), "prediction08": np.around(list(testPredict[7:8].reshape(-1)), decimals=2), "prediction09": np.around(list(testPredict[8:9].reshape(-1)), decimals=2), "prediction10": np.around(list(testPredict[-1:].reshape(-1)), decimals=2), "test_price01": np.around(list(arr2[0:1].reshape(-1)), decimals=2), "test_price02": np.around(list(arr2[1:2].reshape(-1)), decimals=2), "test_price03": np.around(list(arr2[2:3].reshape(-1)), decimals=2), "test_price04": np.around(list(arr2[3:4].reshape(-1)), decimals=2), "test_price05": np.around(list(arr2[4:5].reshape(-1)), decimals=2), "test_price06": np.around(list(arr2[5:6].reshape(-1)), decimals=2), "test_price07": np.around(list(arr2[6:7].reshape(-1)), decimals=2), "test_price08": np.around(list(arr2[7:8].reshape(-1)), decimals=2), "test_price09": np.around(list(arr2[8:9].reshape(-1)), decimals=2), "test_price10": np.around(list(arr2[-1:].reshape(-1)), decimals=2), "volume": np.around(list(train_volume_dataset.reshape(-1)), decimals=2), "entry_test_price": np.around(list(testXArr.reshape(-1)), decimals=2)})
file_name = "sequencePred_1day_for_all_2018_may_2019.csv" 
df.to_csv(file_name, sep=';', index=None)

step = 1
trades_count = 1
for i in range(1733+step, len(dataset)-10, step):
	train_size = i
	dataset_len = len(dataset) 
	# print(len(dataset))
	test_size = len(dataset) - train_size + look_back
	# Need to keep track of volume data in case we include it in price prediction as an input for future cases(added -1 in each index)
	train, test, train_volume_dataset, test_volume_dataset = dataset[train_size-look_back-(forecastCandle+1+step)-9 - (sequenceNumber - 1):train_size,:], dataset[train_size - look_back - (forecastCandle+1):train_size + (forecastCandle+1) + (sequenceNumber - 1),:], df2_volume[train_size-look_back-(forecastCandle+1+step)-9 - (sequenceNumber - 1):train_size,:], df2_volume[train_size - look_back - (forecastCandle+1):train_size + (forecastCandle+1) + (sequenceNumber - 1),:]
	train_timeStamp = df3_timeStamp[train_size-look_back-(forecastCandle+1+step)-9 - (sequenceNumber - 1):train_size,:]
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
	print(trainYArr)
	trainYArr = trainYArr.reshape(-1, 10)
	print('trainYArrAfteReshapestart', trainYArr[0:10])
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
	testYArr = testYArr.reshape(-1, 10)
	testYArr = scaler.inverse_transform(testYArr)
	print('testYArr', testYArr)

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
	model.fit(trainX, trainY, epochs=64, batch_size=60, verbose=1)

	# make predictions
	trainPredict = model.predict(trainX)
	testPredict = model.predict(testX)

	# invert predictions
	trainPredict = scaler.inverse_transform(trainPredict)
	trainY = scaler.inverse_transform(trainY)
	testPredict = scaler.inverse_transform(testPredict)
	print('testPredict', testPredict)
	testY = scaler.inverse_transform(testY)

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
	# trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
	# print('Train Score: %.2f RMSE' % (trainScore))
	# testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
	# print('Test Score: %.2f RMSE' % (testScore))

	# shift train predictions for plotting
	trainPredictPlot = np.empty_like(dataset)
	trainPredictPlot[:, :] = np.nan
	# trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

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
	# trainY = trainY.reshape(-1, 1)
	# trainY = trainY[-1:]
	testPredict = testPredict.reshape(-1, 1)
	arr2 = arr2.reshape(-1, 1)
	print('testXArr', testXArr)
	testXArr = testXArr[-1:]
	print('testXArr', testXArr)

	train_volume_dataset = train_volume_dataset[-1:]
	print('train_volume_dataset', train_volume_dataset)
	train_timeStamp = train_timeStamp[-2:-1]
	print('train_timeStamp', train_timeStamp[-1:])

	train_volume_dataset = train_volume_dataset[-1:]
	# print('test_rsi_dataset', test_rsi_dataset)
	# callTakingProb = nn2_for_1hr.predict_value(trainY, testPredict, train_volume_dataset)
	# upDownPred = lstm_up_down_step_01.predict_upDown(test_rsi_dataset)
	# if testPredict[len(testPredict)-1][0] > trainY[len(trainY)-1][0]:
	#     actionTaken = 1
	# else:
	#     actionTaken = 0
	# print('nn2_appending_inputs', trainY[0][0], testPredict[0][0], actionTaken, arr2[0][0], train_volume_dataset[0][0])
	# nn2_for_1hr.appendLatestTradeExample(trainY[0][0], testPredict[0][0], actionTaken, arr2[0][0], train_volume_dataset[0][0])
	# trades_count = trades_count+1
	# if trades_count == 10:
	#     nn2_for_1hr.retrainingNN2()
	#     trades_count = 0
	# print('callTakingProb', callTakingProb)
	# export prediction and actual prices
	
	df = pd.DataFrame(data={"timeStamp": np.around(list(train_timeStamp[-1].reshape(-1)), decimals=2), "prediction01": np.around(list(testPredict[0:1].reshape(-1)), decimals=2), "prediction02": np.around(list(testPredict[1:2].reshape(-1)), decimals=2), "prediction03": np.around(list(testPredict[2:3].reshape(-1)), decimals=2), "prediction04": np.around(list(testPredict[3:4].reshape(-1)), decimals=2), "prediction05": np.around(list(testPredict[4:5].reshape(-1)), decimals=2), "prediction06": np.around(list(testPredict[5:6].reshape(-1)), decimals=2), "prediction07": np.around(list(testPredict[6:7].reshape(-1)), decimals=2), "prediction08": np.around(list(testPredict[7:8].reshape(-1)), decimals=2), "prediction09": np.around(list(testPredict[8:9].reshape(-1)), decimals=2), "prediction10": np.around(list(testPredict[-1:].reshape(-1)), decimals=2), "test_price01": np.around(list(arr2[0:1].reshape(-1)), decimals=2), "test_price02": np.around(list(arr2[1:2].reshape(-1)), decimals=2), "test_price03": np.around(list(arr2[2:3].reshape(-1)), decimals=2), "test_price04": np.around(list(arr2[3:4].reshape(-1)), decimals=2), "test_price05": np.around(list(arr2[4:5].reshape(-1)), decimals=2), "test_price06": np.around(list(arr2[5:6].reshape(-1)), decimals=2), "test_price07": np.around(list(arr2[6:7].reshape(-1)), decimals=2), "test_price08": np.around(list(arr2[7:8].reshape(-1)), decimals=2), "test_price09": np.around(list(arr2[8:9].reshape(-1)), decimals=2), "test_price10": np.around(list(arr2[-1:].reshape(-1)), decimals=2), "volume": np.around(list(train_volume_dataset.reshape(-1)), decimals=2), "entry_test_price": np.around(list(testXArr.reshape(-1)), decimals=2)})
	#file_name = "lstm_result_5min_x_is_10_retraining2"+ str(train_size)+ ".csv" 
	df.to_csv(file_name, sep=';', mode = 'a', index=None, header=None)

	# plot the actual price, prediction in test data=red line, actual price=blue line
	#plt.plot(testPredictPlot)
	#plt.show()

