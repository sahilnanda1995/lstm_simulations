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

input_file="nn2examples.csv"

entry_test_price = read_csv(input_file, index_col=None, header=None, delimiter=';', usecols=[0])
prediction = read_csv(input_file, index_col=None, header=None, delimiter=';', usecols=[1])
test_price = read_csv(input_file, index_col=None, header=None, delimiter=';', usecols=[2])
volume = read_csv(input_file, index_col=None, header=None, delimiter=';', usecols=[3])

arr = []

entry_test_price = entry_test_price.values
prediction = prediction.values
test_price = test_price.values
volume = volume.values

scaler_volume = MinMaxScaler(feature_range=(0, 1))

scaler_predProfit = MinMaxScaler(feature_range=(0, 1))

arrX = []
arrY = []
actionArr = []
predProfitArr = []
volumeArr = []
volumeRetrainArr = []

for i in range(0,len(volume),10):
    volumeArr.append(volume[i][0])

volumeRetrainArr = volumeArr
volumeArr = np.array(volumeArr)
volumeArr = volumeArr.reshape(-1, 1)
volumeArr = scaler_volume.fit_transform(volumeArr)

for i in range(0,len(entry_test_price),10):
    predProfit = float(prediction[i][0] - entry_test_price[i][0])/entry_test_price[i][0] * 100
    # print('predicted profit', predProfit)
    if predProfit >= 0:
        action = 1
    else:
        action = 0
        predProfit = abs(predProfit)
    predProfitArr.append(predProfit)
    actionArr.append(action)
    actualProfit = float(test_price[i][0] - entry_test_price[i][0])/entry_test_price[i][0] * 100
    # print('actual profit', actualProfit)
    if action == 1:
        if actualProfit >= 0.2:
            arrY.append(1)
        else:
            arrY.append(0)
    elif action == 0:
        if actualProfit <= -0.2:
            arrY.append(1)
        else:
            arrY.append(0)


print('volume length', len(volume))

trainYArr = predProfitArr
actionRetrainArr = actionArr
yLabel = arrY



predProfitArr = np.array(predProfitArr)
predProfitArr = predProfitArr.reshape(-1,1)
predProfitArr = scaler_predProfit.fit_transform(predProfitArr)
# print(predProfitArr)
actionArr = np.array(actionArr)
actionArr = actionArr.reshape(-1,1)
print(len(volumeArr))
arrY = np.array(arrY)
arrY = arrY.reshape(-1,1)
print(arrY)
print(len(arrY))

for i in range(len(predProfitArr)):
    arrX.append(predProfitArr[i][0])
    arrX.append(actionArr[i][0])
    arrX.append(volumeArr[i][0])

print(len(arrX))
# print(arrX)

arrX = np.array(arrX)
arrX = arrX.reshape(-1,3)

model = Sequential()
model.add(Dense(25, input_dim=3, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(arrX, arrY, epochs=64, verbose=1)
# new instances where we do not know the answer

ynew = model.predict_proba(arrX[0:30])

print(len(trainYArr), len(actionRetrainArr), len(volumeArr), len(yLabel))

# model.save('notSkipping_nn2.h5')  # creates a HDF5 file 'notSkipping_nn2.h5'


def appendLatestTradeExample(previous_price, previous_predictedPrice, actionTaken, actualPrice, volume):
    global trainYArr
    global actionRetrainArr
    global volumeRetrainArr
    global yLabel
    print('predictedPrice', previous_predictedPrice, 'previousPrice', previous_price, 'actualPrice', actualPrice)
    trainYArr.append(abs(float(previous_predictedPrice - previous_price))/previous_price*100)
    actionRetrainArr.append(actionTaken)
    volumeRetrainArr.append(volume)
    profit = float(actualPrice-previous_price)/previous_price*100
    if actionTaken == 0:
        profit = -profit
    if profit >= 0.2 :
        yLabel.append(1)
    else:
        yLabel.append(0)

    print(len(trainYArr), len(actionRetrainArr), len(volumeRetrainArr), len(yLabel))



# function to retrain NN2 with the new examples
def retrainingNN2():
    arrXRetrain = []
    global trainYArr
    global actionRetrainArr
    global volumeRetrainArr
    global yLabel
    print(len(trainYArr), len(actionRetrainArr), len(volumeRetrainArr))
    # print('trainYArr', trainYArr)
    # print('actionRetrainArr', actionRetrainArr)
    # print('volumeRetrainArr', volumeRetrainArr)
    # print('yLabel', yLabel)
    trainYArr2 = np.array(trainYArr)
    trainYArr2 = trainYArr2.reshape(-1, 1)
    trainYArr2 = scaler_predProfit.transform(trainYArr2)
    actionRetrainArr2 = np.array(actionRetrainArr)
    actionRetrainArr2 = actionRetrainArr2.reshape(-1, 1)
    volumeRetrainArr2 = np.array(volumeRetrainArr)
    volumeRetrainArr2 = volumeRetrainArr2.reshape(-1, 1)
    volumeRetrainArr2 = scaler_volume.transform(volumeRetrainArr2)
    yLabel2 = np.array(yLabel)
    yLabel2 = yLabel2.reshape(-1, 1)
    #print('volumeRetrainArr', volumeRetrainArr)
    for i in range(len(trainYArr)):
        arrXRetrain.append(trainYArr2[i][0])
        arrXRetrain.append(actionRetrainArr2[i][0])
        arrXRetrain.append(volumeRetrainArr2[i])
    arrXRetrain = np.array(arrXRetrain)
    arrXRetrain = arrXRetrain.reshape(-1, 3)
    print(arrXRetrain)
    # model = load_model('notSkipping_nn2.h5')
    print('nn2retraining')
    model.fit(arrXRetrain, yLabel, epochs=64, verbose=1)
    # model.save('notSkipping_nn2.h5')
    # trainYArr = []
    # actionRetrainArr = []
    # volumeRetrainArr = []
    # yLabel = []
    # print(len(trainYArr), len(actionRetrainArr), len(volumeRetrainArr))


#function to predict the probability of NN2 for not skipping a trade
def predict_value(trainY, prediction, volumeX):
    volumeX = scaler_volume.transform(volumeX)
    predProfitX = []
    actionX = []
    trainX = []
    for i in range(len(trainY)):
        predProfit = float(prediction[i][0] - trainY[i][0])/trainY[i][0] * 100
        # print('predicted profit', predProfit)
        if predProfit >= 0:
            action = 1
        else:
            action = 0
            predProfit = abs(predProfit)
        predProfitX.append(predProfit)
        actionX.append(action)
    # print(predProfitX)
    # print(actionX)
    actionX = np.array(actionX)
    actionX = actionX.reshape(-1, 1)
    predProfitX = np.array(predProfitX)
    predProfitX = predProfitX.reshape(-1, 1)
    predProfitX = scaler_predProfit.transform(predProfitX)
    for i in range(len(predProfitX)):
        trainX.append(predProfitX[i][0])
        trainX.append(actionX[i][0])
        trainX.append(volumeX[i][0])
    # print(trainX)
    trainX = np.array(trainX)
    trainX = trainX.reshape(-1,3)
    # print(trainX)
    #model = load_model('notSkipping_nn2.h5')
    predProb = model.predict_proba(trainX)
    # print('inside predval', predProb)
    return predProb
