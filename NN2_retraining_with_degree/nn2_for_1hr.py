# example making new probability predictions for a classification problem
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

input_file="bit_1hr_trade_examples_for_nn2.csv"

entry_test_price = read_csv(input_file, index_col=None, header=None, delimiter=';', usecols=[0])
prediction = read_csv(input_file, index_col=None, header=None, delimiter=';', usecols=[1])
test_price = read_csv(input_file, index_col=None, header=None, delimiter=';', usecols=[2])
volume = read_csv(input_file, index_col=None, header=None, delimiter=';', usecols=[3])


entry_test_price = entry_test_price.values
prediction = prediction.values
test_price = test_price.values
volume = volume.values

scaler_volume = MinMaxScaler(feature_range=(0, 1))

#volume = scaler_volume.fit_transform(volume)

scaler_predProfit = MinMaxScaler(feature_range=(0, 1))

arrX = []
arrY = []
actionArr = []
predProfitArr = []
volumeArr = []

for i in range(0,len(volume)):
    volumeArr.append(volume[i][0])

for i in range(0,len(entry_test_price)):
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
    numberOfClasses = 18
    y_label = []
    for i in range(numberOfClasses):
        y_label.append(0)
    if action == 0:
        actualProfit = -actualProfit
    if actualProfit < -2:
        y_label[0] = 1
    elif actualProfit >= -2 and actualProfit < -1:
        y_label[1] = 1
    elif actualProfit >= -1 and actualProfit < -0.5:
        y_label[2] = 1
    elif actualProfit >= -0.5 and actualProfit < -0.3:
        y_label[3] = 1
    elif actualProfit >= -0.3 and actualProfit < -0.1:
        y_label[4] = 1
    elif actualProfit >= -0.1 and actualProfit < 0:
        y_label[5] = 1
    elif actualProfit >= 0 and actualProfit < 0.1:
        y_label[6] = 1
    elif actualProfit >= 0.1 and actualProfit < 0.2:
        y_label[7] = 1
    elif actualProfit >= 0.2 and actualProfit < 0.3:
        y_label[8] = 1
    elif actualProfit >= 0.3 and actualProfit < 0.4:
        y_label[9] = 1
    elif actualProfit >= 0.4 and actualProfit < 0.6:
        y_label[10] = 1
    elif actualProfit >= 0.6 and actualProfit < 0.8:
        y_label[11] = 1
    elif actualProfit >= 0.8 and actualProfit < 1.0:
        y_label[12] = 1
    elif actualProfit >= 1.0 and actualProfit < 1.5:
        y_label[13] = 1
    elif actualProfit >= 1.5 and actualProfit < 2.0:
        y_label[14] = 1
    elif actualProfit >= 2.0 and actualProfit < 3.0:
        y_label[15] = 1
    elif actualProfit >= 3.0 and actualProfit < 5.0:
        y_label[16] = 1
    elif actualProfit >= 5:
        y_label[17] = 1
    arrY.append(y_label)


print('arrY', arrY)
predProfitArr = np.array(predProfitArr)
predProfitArr = predProfitArr.reshape(-1,1)
predProfitArr = scaler_predProfit.fit_transform(predProfitArr)
print(predProfitArr[0:10])

volumeArr = np.array(volume)
volumeArr = volumeArr.reshape(-1,1)
volumeArr = scaler_volume.fit_transform(volumeArr)
print(volumeArr[0:10])

actionArr = np.array(actionArr)
actionArr = actionArr.reshape(-1,1)
print(len(volumeArr))
arrY = np.array(arrY)
arrY = arrY.reshape(-1,18)
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
# print(arrX)
# print(X)
# print(y)
# scalar = MinMaxScaler()
# scalar.fit(X)
# print(X)
# X = scalar.transform(X)

# X = [0.89,0.1,0.95,0.3,0.79,0.2,0.12,0.7,0.33,0.9,0.22,0.89]
# X = np.array(X)
# X = X.reshape(-1,2)
# y = [1,1,1,0,0,0]
# y = np.array(y)



# print(X)
# define and fit the final model


model = Sequential()
model.add(Dense(25, input_dim=3, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(18, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(arrX, arrY, epochs=64, verbose=1)
# new instances where we do not know the answer

ynew = model.predict_classes(arrX[0:150])

for i in range(0,150):
    print("X=%s, Predicted=%s, arrY=%s" % (arrX[i], ynew[i], arrY[i]))


# model.save('notSkipping_nn2_1hr.h5')  # creates a HDF5 file 'notSkipping_nn2_1hr.h5'

trainYArr = []
actionRetrainArr = []
yLabel = []
volumeRetrainArr = []

def appendLatestTradeExample(previous_price, previous_predictedPrice, actionTaken, actualPrice, volume):
    global trainYArr
    global actionRetrainArr
    global volumeRetrainArr
    global yLabel
    print('predictedPrice', previous_predictedPrice, 'previousPrice', previous_price, 'actualPrice', actualPrice, 'volume', volume)
    trainYArr.append([abs(float(previous_predictedPrice - previous_price))/previous_price*100])
    actionRetrainArr.append([actionTaken])
    volumeRetrainArr.append([volume])
    actualProfit = float(actualPrice-previous_price)/previous_price*100
    if actionTaken == 0:
        actualProfit = -actualProfit
    numberOfClasses = 18
    y_label = []
    for i in range(numberOfClasses):
        y_label.append(0)
    if action == 0:
        actualProfit = -actualProfit
    if actualProfit < -2:
        y_label[0] = 1
    elif actualProfit >= -2 and actualProfit < -1:
        y_label[1] = 1
    elif actualProfit >= -1 and actualProfit < -0.5:
        y_label[2] = 1
    elif actualProfit >= -0.5 and actualProfit < -0.3:
        y_label[3] = 1
    elif actualProfit >= -0.3 and actualProfit < -0.1:
        y_label[4] = 1
    elif actualProfit >= -0.1 and actualProfit < 0:
        y_label[5] = 1
    elif actualProfit >= 0 and actualProfit < 0.1:
        y_label[6] = 1
    elif actualProfit >= 0.1 and actualProfit < 0.2:
        y_label[7] = 1
    elif actualProfit >= 0.2 and actualProfit < 0.3:
        y_label[8] = 1
    elif actualProfit >= 0.3 and actualProfit < 0.4:
        y_label[9] = 1
    elif actualProfit >= 0.4 and actualProfit < 0.6:
        y_label[10] = 1
    elif actualProfit >= 0.6 and actualProfit < 0.8:
        y_label[11] = 1
    elif actualProfit >= 0.8 and actualProfit < 1.0:
        y_label[12] = 1
    elif actualProfit >= 1.0 and actualProfit < 1.5:
        y_label[13] = 1
    elif actualProfit >= 1.5 and actualProfit < 2.0:
        y_label[14] = 1
    elif actualProfit >= 2.0 and actualProfit < 3.0:
        y_label[15] = 1
    elif actualProfit >= 3.0 and actualProfit < 5.0:
        y_label[16] = 1
    elif actualProfit >= 5:
        y_label[17] = 1
    yLabel.append(y_label)
    print(len(trainYArr), len(actionRetrainArr), len(volumeRetrainArr))



# function to retrain NN2 with the new examples
def retrainingNN2():
    arrXRetrain = []
    global trainYArr
    global actionRetrainArr
    global volumeRetrainArr
    global yLabel
    print(len(trainYArr), len(actionRetrainArr), len(volumeRetrainArr))
    print('trainYArr', trainYArr)
    print('actionRetrainArr', actionRetrainArr)
    print('volumeRetrainArr', volumeRetrainArr)
    print('yLabel', yLabel)
    trainYArr = np.array(trainYArr)
    trainYArr = trainYArr.reshape(-1, 1)
    trainYArr = scaler_predProfit.transform(trainYArr)
    actionRetrainArr = np.array(actionRetrainArr)
    actionRetrainArr = actionRetrainArr.reshape(-1, 1)
    volumeRetrainArr = np.array(volumeRetrainArr)
    volumeRetrainArr = volumeRetrainArr.reshape(-1, 1)
    volumeRetrainArr = scaler_volume.transform(volumeRetrainArr)
    yLabel = np.array(yLabel)
    yLabel = yLabel.reshape(-1, 18)
    #print('volumeRetrainArr', volumeRetrainArr)
    for i in range(len(trainYArr)):
        arrXRetrain.append(trainYArr[i][0])
        arrXRetrain.append(actionRetrainArr[i][0])
        arrXRetrain.append(volumeRetrainArr[i][0])
    arrXRetrain = np.array(arrXRetrain)
    arrXRetrain = arrXRetrain.reshape(-1, 3)
    print('arrXRetrain', arrXRetrain, 'yLabel', yLabel)
    # model = load_model('notSkipping_nn2_1hr.h5')
    model.fit(arrXRetrain, yLabel, epochs=64, verbose=1)
    # model.save('notSkipping_nn2_1hr.h5')
    trainYArr = []
    actionRetrainArr = []
    volumeRetrainArr = []
    yLabel = []
    print(len(trainYArr), len(actionRetrainArr), len(volumeRetrainArr))


#function to predict the probability of NN2 for not skipping a trade
def predict_value(trainY, prediction, volumeX):
    # print('Inside predict_value')
    # print(trainY)
    # print(prediction)
    # print(volumeX)
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
    # model = load_model('notSkipping_nn2_1hr.h5')
    predProb = model.predict_classes(trainX)
    # print('inside predval', predProb)
    return predProb
