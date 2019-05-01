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
input_file="1yr1hr_step_10_01_02_05.csv"

forecastCandle = 4
# convert an array of values into a dataset matrix
def create_dataset(dataset, y_label, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1-forecastCandle):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(y_label[i + look_back + forecastCandle, 0])
	return np.array(dataX), np.array(dataY)

# fix random seed for reproducibility
np.random.seed(5)

# load the dataset
df = read_csv(input_file, header=None, index_col=None, delimiter=',', usecols=[6])
df_upDown = read_csv(input_file, header=None, index_col=None, delimiter=',', usecols=[10])
# take close price column[5]
all_x = df.values
dataset=all_x.reshape(-1, 1)

# up or down dataset
y_dataset = df_upDown.values
y_dataset = y_dataset.reshape(-1, 1)

# dataset length
print(len(dataset))
print(len(y_dataset))

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

look_back = 30
# split into train and test sets, 50% test data, 50% training data
#size of 1 year data
train_size = 8760
dataset_len = len(dataset) 
print(len(dataset))
test_size = len(dataset) - train_size + look_back
train, test = dataset[0:train_size,:], dataset[train_size - look_back - (forecastCandle+1):,:]
y_train, y_test = y_dataset[0:train_size,:], y_dataset[train_size - look_back - (forecastCandle+1):,:]

# reshape into X=t and Y=t+1, timestep 240
trainX, trainY = create_dataset(train, y_train, look_back)
testX, testY = create_dataset(test, y_test, look_back)

print(trainX[-10:])

for i in range(10):
    trainXArr = []
    for val in trainX[len(trainX)-(10 - i)]:
        trainXArr.append(val)

    trainXArr = np.array(trainXArr)
    trainXArr = trainXArr[-10:]
    trainXArr = trainXArr.reshape(-1,1)
    # print(trainXArr)
    trainXArr = scaler.inverse_transform(trainXArr)

    print('trainXArr', trainXArr)

trainYArr = trainY[-10:]
trainYArr = np.array(trainYArr)
trainYArr = trainYArr.reshape(-1, 1)
# trainYArr = scaler.inverse_transform(trainYArr)
print('trainYArr', trainYArr)

for i in range(10):
    testXArr = []
    for val in testX[len(testX)-(10 - i)]:
        testXArr.append(val)


    testXArr = np.array(testXArr)
    testXArr = testXArr[-10:]
    testXArr = testXArr.reshape(-1,1)
    # print(testXArr)
    testXArr = scaler.inverse_transform(testXArr)

    print('testXArr', testXArr)

testYArr = testY[-10:]
testYArr = np.array(testYArr)
testYArr = testYArr.reshape(-1, 1)
# testYArr = scaler.inverse_transform(testYArr)
print('testYArr', testYArr)
print('len testYArr', len(testYArr))


# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the LSTM network, optimizer=adam, 25 neurons, dropout 0.1
model = Sequential()
model.add(LSTM(200, dropout=0.2, recurrent_dropout=0.2, input_shape=(1, look_back)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model.summary())
model.fit(trainX, trainY, epochs=20, batch_size=5)
# Final evaluation of the model
scores = model.evaluate(testX, testY, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
# make predictions
trainPredict = model.predict_classes(trainX)
testPredict = model.predict_classes(testX)

# print(history.history.keys())
# # summarize history for accuracy
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
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


# invert predictions
# trainPredict = scaler.inverse_transform(trainPredict)
# trainY = scaler.inverse_transform([trainY])
# testPredict = scaler.inverse_transform(testPredict)
# testY = scaler.inverse_transform([testY])


# calculate root mean squared error
# trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
# print('Train Score: %.2f RMSE' % (trainScore))
# testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
# print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
# trainPredictPlot = np.empty_like(dataset)
# trainPredictPlot[:, :] = np.nan
# trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting
# testPredictPlot = np.empty_like(dataset)
# testPredictPlot[:, :] = np.nan

arr2 = testY
arr2 = np.array(arr2)
arr2 = arr2.reshape(-1, 1)
print('arr2', arr2)
print(len(arr2))
# trainY = trainY.reshape(-1, 1)
# trainY = trainY[-1:]
# arr2 = arr2[-1:]
# testPredict = testPredict[-1:]
# print('trainY2', trainY)
print('testPredictions:')
print(testPredict)
print(len(testPredict))

# export prediction and actual prices
df = pd.DataFrame(data={"prediction": np.around(list(testPredict.reshape(-1)), decimals=2), "test_price": np.around(list(arr2.reshape(-1)), decimals=2)})
file_name = "lstm_upDown_step_05_result.csv" 
df.to_csv(file_name, sep=';', index=None)
#df.to_json("testJson.json", orient = 'records')

# plot the actual price, prediction in test data=red line, actual price=blue line
#plt.plot(testPredictPlot)
#plt.show()









# step = 10
# for i in range(105121+step, dataset_len - step, step):
#     train_size = i
#     dataset_len = len(dataset) 
#     print(len(dataset))
#     test_size = len(dataset) - train_size + look_back
#     train, test = dataset[train_size-look_back-(forecastCandle+1+step):train_size,:], dataset[train_size - look_back - (forecastCandle+1):train_size + (forecastCandle+1),:]

#     # reshape into X=t and Y=t+1, timestep 240
#     trainX, trainY = create_dataset(train, look_back)
#     testX, testY = create_dataset(test, look_back)

#     trainXArr = []
#     for val in trainX[len(trainX)-1]:
#         trainXArr.append(val)

#     trainXArr = np.array(trainXArr)
#     trainXArr = trainXArr[-10:]
#     trainXArr = trainXArr.reshape(-1,1)
#     # print(trainXArr)
#     trainXArr = scaler.inverse_transform(trainXArr)
#     print('trainXArr', trainXArr)

#     trainYArr = trainY
#     trainYArr = np.array(trainYArr)
#     trainYArr = trainYArr.reshape(-1, 1)
#     trainYArr = scaler.inverse_transform(trainYArr)
#     print('trainYArr', trainYArr)

#     testXArr = []
#     for val in testX[len(testX)-1]:
#         testXArr.append(val)


#     testXArr = np.array(testXArr)
#     testXArr = testXArr[-10:]
#     testXArr = testXArr.reshape(-1,1)
#     print(testXArr)
#     testXArr = scaler.inverse_transform(testXArr)
#     print('testXArr', testXArr)

#     testYArr = testY
#     testYArr = np.array(testYArr)
#     testYArr = testYArr.reshape(-1, 1)
#     testYArr = scaler.inverse_transform(testYArr)
#     print('testYArr', testYArr)

    
#     # reshape input to be [samples, time steps, features]
#     trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
#     testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

#     # create and fit the LSTM network, optimizer=adam, 25 neurons, dropout 0.1
#     #model = Sequential()
#     #model.add(LSTM(25, input_shape=(1, look_back)))
#     #model.add(Dropout(0.1))
#     #model.add(Dense(1))
#     #model.compile(loss='mse', optimizer='adam')
#     model.fit(trainX, trainY, epochs=64, batch_size=60, verbose=1)

#     # make predictions
#     trainPredict = model.predict(trainX)
#     testPredict = model.predict(testX)

#     # invert predictions
#     trainPredict = scaler.inverse_transform(trainPredict)
#     trainY = scaler.inverse_transform([trainY])
#     testPredict = scaler.inverse_transform(testPredict)
#     testY = scaler.inverse_transform([testY])

#     trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
#     print('Train Score: %.2f RMSE' % (trainScore))
#     testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
#     print('Test Score: %.2f RMSE' % (testScore))

#     # shift train predictions for plotting
#     trainPredictPlot = np.empty_like(dataset)
#     trainPredictPlot[:, :] = np.nan
#     trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

#     # shift test predictions for plotting
#     testPredictPlot = np.empty_like(dataset)
#     testPredictPlot[:, :] = np.nan
    
#     arr2 = testYArr
#     print('arr2', arr2)

#     trainY = trainY.reshape(-1, 1)
#     trainY = trainY[-1:]
#     arr2 = arr2[-1:]
#     testPredict = testPredict[-1:]
#     print('trainY2', trainY)
#     print('testPredictions:')
#     print(testPredict)
#     print(len(testPredict))

#     # export prediction and actual prices
#     df = pd.DataFrame(data={"prediction": np.around(list(testPredict.reshape(-1)), decimals=2), "test_price": np.around(list(arr2.reshape(-1)), decimals=2), "entry_test_price": np.around(list(trainY.reshape(-1)), decimals=2)})
#     #file_name = "lstm_result_5min_x_is_10_retraining2"+ str(train_size)+ ".csv" 
#     df.to_csv(file_name, sep=';', mode = 'a', index=None, header = None)

#     # plot the actual price, prediction in test data=red line, actual price=blue line
#     #plt.plot(testPredictPlot)
#     #plt.show()

