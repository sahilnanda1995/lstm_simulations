# example making new probability predictions for a classification problem
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

input_file="example_logs_for_nn2.csv"

priceData = read_csv(input_file, index_col=None, header=0, delimiter=';', usecols=[0, 1, 2, 3, 4, 5])
print(priceData)

priceData = priceData.values

dataY = []

for i in range(len(priceData)):
    # print(priceData[i])
    if priceData[i][2] - priceData[i][1] > priceData[i][1] - priceData[i][3]:
        dataY.append(1)
    else:
        dataY.append(0)

print(len(dataY))

for i in range(len(dataY)):
    print(dataY[i])

priceData = np.array(priceData)

priceData =  priceData.reshape(-1, 1)
print(priceData)

scaler_price = MinMaxScaler(feature_range=(0, 1))

priceData = scaler_price.fit_transform(priceData)

priceData = priceData.reshape(-1, 6)

for i in range(len(priceData)):
    print(priceData[i])

volumeData = read_csv(input_file, index_col=None, header=0, delimiter=';', usecols=[9])
print(volumeData)
volumeData = volumeData.values

scaler_volume = MinMaxScaler(feature_range=(0, 1))

volumeData = scaler_volume.fit_transform(volumeData)

print(len(volumeData))


dataX = []

for i in range(len(priceData)):
    # print(priceData[i])
    dataX.append([priceData[i][1], priceData[i][4], priceData[i][5], volumeData[i][0]])

for i in range(len(dataX)):
    print(dataX[i])

print(len(dataX))

dataX = np.array(dataX)
dataY = np.array(dataY)


model = Sequential()
model.add(Dense(25, input_dim=4, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(dataX, dataY, epochs=64, verbose=1)
# # new instances where we do not know the answer

ynew = model.predict_classes(dataX[0:730])


for i in range(0,730):
    print("X=%s, Predicted=%s, arrY=%s" % (dataX[i], ynew[i], dataY[i]))



def predict_action(entryPrice, highPred, lowPred, vol):
    arrX = []
    entryPrice = scaler_price.transform([[entryPrice]])
    highPred = scaler_price.transform([[highPred]])
    lowPred = scaler_price.transform([[lowPred]])
    vol = scaler_volume.transform([[vol]])
    arrX.append([entryPrice[0][0], highPred[0][0], lowPred[0][0], vol[0][0]])
    print('arrX', arrX)
    arrX = np.array(arrX)
    print('arrX', arrX)
    action = model.predict_classes(arrX)
    return action


print(predict_action(981.78, 797.489990, 668.530029, 11661.46))



# model.save('notSkipping_nn2.h5')  # creates a HDF5 file 'notSkipping_nn2.h5'

# trainYArr = []
# actionRetrainArr = []
# yLabel = []
# volumeRetrainArr = []

# def appendLatestTradeExample(previous_price, previous_predictedPrice, actionTaken, actualPrice, volume):
#     global trainYArr
#     global actionRetrainArr
#     global volumeRetrainArr
#     global yLabel
#     print('predictedPrice', previous_predictedPrice, 'previousPrice', previous_price, 'actualPrice', actualPrice)
#     trainYArr.append([abs(float(previous_predictedPrice - previous_price))/previous_price*100])
#     actionRetrainArr.append([actionTaken])
#     volumeRetrainArr.append([volume])
#     profit = float(actualPrice-previous_price)/previous_price*100
#     if actionTaken == 0:
#         profit = -profit
#     if profit >= 0.2 :
#         yLabel.append([1])
#     else:
#         yLabel.append([0])

#     print(len(trainYArr), len(actionRetrainArr), len(volumeRetrainArr))



# # function to retrain NN2 with the new examples
# def retrainingNN2():
#     arrXRetrain = []
#     global trainYArr
#     global actionRetrainArr
#     global volumeRetrainArr
#     global yLabel
#     print(len(trainYArr), len(actionRetrainArr), len(volumeRetrainArr))
#     print('trainYArr', trainYArr)
#     print('actionRetrainArr', actionRetrainArr)
#     print('volumeRetrainArr', volumeRetrainArr)
#     print('yLabel', yLabel)
#     trainYArr = np.array(trainYArr)
#     trainYArr = trainYArr.reshape(-1, 1)
#     trainYArr = scaler_predProfit.transform(trainYArr)
#     actionRetrainArr = np.array(actionRetrainArr)
#     actionRetrainArr = actionRetrainArr.reshape(-1, 1)
#     volumeRetrainArr = np.array(volumeRetrainArr)
#     volumeRetrainArr = volumeRetrainArr.reshape(-1, 1)
#     volumeRetrainArr = scaler_volume.transform(volumeRetrainArr)
#     yLabel = np.array(yLabel)
#     yLabel = yLabel.reshape(-1, 1)
#     #print('volumeRetrainArr', volumeRetrainArr)
#     for i in range(len(trainYArr)):
#         arrXRetrain.append(trainYArr[i][0])
#         arrXRetrain.append(actionRetrainArr[i][0])
#         arrXRetrain.append(volumeRetrainArr[i])
#     arrXRetrain = np.array(arrXRetrain)
#     arrXRetrain = arrXRetrain.reshape(-1, 3)
#     # print(arrXRetrain)
#     # model = load_model('notSkipping_nn2.h5')
#     print('nn2Retraining')
#     model.fit(arrXRetrain, yLabel, epochs=64, verbose=1)
#     # model.save('notSkipping_nn2.h5')
#     trainYArr = []
#     actionRetrainArr = []
#     volumeRetrainArr = []
#     yLabel = []
#     print(len(trainYArr), len(actionRetrainArr), len(volumeRetrainArr))


# #function to predict the probability of NN2 for not skipping a trade
# def predict_value(trainY, prediction, volumeX):
#     # print('Inside predict_value')
#     # print(trainY)
#     # print(prediction)
#     # print(volumeX)
#     volumeX = scaler_volume.transform(volumeX)
#     predProfitX = []
#     actionX = []
#     trainX = []
#     for i in range(len(trainY)):
#         predProfit = float(prediction[i][0] - trainY[i][0])/trainY[i][0] * 100
#         # print('predicted profit', predProfit)
#         if predProfit >= 0:
#             action = 1
#         else:
#             action = 0
#             predProfit = abs(predProfit)
#         predProfitX.append(predProfit)
#         actionX.append(action)
#     # print(predProfitX)
#     # print(actionX)
#     actionX = np.array(actionX)
#     actionX = actionX.reshape(-1, 1)
#     predProfitX = np.array(predProfitX)
#     predProfitX = predProfitX.reshape(-1, 1)
#     predProfitX = scaler_predProfit.transform(predProfitX)
#     for i in range(len(predProfitX)):
#         trainX.append(predProfitX[i][0])
#         trainX.append(actionX[i][0])
#         trainX.append(volumeX[i][0])
#     # print(trainX)
#     trainX = np.array(trainX)
#     trainX = trainX.reshape(-1,3)
#     # print(trainX)
#     # model = load_model('notSkipping_nn2.h5')
#     predProb = model.predict_proba(trainX)
#     # print('inside predval', predProb)
#     return predProb
