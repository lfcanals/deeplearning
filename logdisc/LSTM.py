import time
import keras
import numpy as np
import matplotlib.pyplot as plt
import Bruce

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import RNN 



##
## The model takes a sequence of calculated values Y and inferes the next Y of the sequence
##
# Where
#       x_trains is a list of arrays
#                all arrays of same length
#             
#       x_tests  is a list of arrays
#                of same length as x_trains
#
#       y_tests is a list of numbers
#               which lenght is the number of arrays in x_tests
#                
def run(targetFunction, x_trains, x_tests):
    neuronsPerLayer = 32
    numOfLayers = 2
    model = Sequential()
    for i in range(numOfLayers-1):
        model.add(LSTM(neuronsPerLayer, return_sequences=True, activation=None))
    model.add(LSTM(neuronsPerLayer, activation=None))
    model.add(Dense(1, activation=None))
    
    model.compile(loss='mse', optimizer='adam', metrics=['mean_squared_error'])

    
    
    #
    # Load data in proper format
    #
    lstmDataSet = Bruce.LSTMDataSet()

    y_for_training = []
    for x_train in x_trains:
        y_train = targetFunction(x_train)
        lstmDataSet.addTrainExample(y_train[0:len(y_train)-1], y_train[len(y_train)-1])


    y_tests = []
    for x_test in x_tests:
        y_test = targetFunction(x_test)
        lstmDataSet.addTestExample(y_test[0:len(y_test)-1])
        y_tests.append(y_test[len(y_test)-1])

    lstmDataSet.summary()


    # 
    # Train it
    #
    t0 = time.perf_counter()
    y_trains_previous, y_trains_forecasts = lstmDataSet.getDataSets()
    history = model.fit(y_trains_previous, y_trains_forecasts, epochs=100, verbose=1)
    t1 = time.perf_counter()
    print('Training time :', t1-t0)

    
    #
    # Test it
    #
    y_forecasts = []
    for y in lstmDataSet.getTestSet():
        y_test_forecast = model.predict(y)
        y_forecasts.append(y_test_forecast[0][0])

    plt.plot(y_tests, label='reality')
    plt.plot(y_forecasts, label='forecasting')
    plt.legend(loc='upper left')
    plt.show()


