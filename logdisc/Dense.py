import time
import keras
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import RNN 


###### MAIN CODE 

def run(targetFunction, x_train, x_test):
    # Layers, dense
    neuronsPerLayer = 256
    numOfLayers = 20
    
    model = Sequential()
    model.add(Dense(neuronsPerLayer, input_shape=(1,), activation='relu'))
    for i in range(numOfLayers-1):
        model.add(Dense(neuronsPerLayer, activation='relu'))

    model.add(Dense(1))
    
    model.compile(loss='mse', optimizer='SGD', metrics=['mean_squared_error'])
    
    
    # 
    # Train it
    #

    # Train set
    y_train = targetFunction(x_train)   # Normalize between 0 and 1, as relu only works here
    
    #x_train = x_train.reshape(x_train.shape[0],1,1) # For LSTM only
    
    # Train process
    t0 = time.perf_counter()
    history = model.fit(x_train, y_train, epochs=1000, verbose=0)
    t1 = time.perf_counter()
    print('Training time :', t1-t0)
    
    
    # Predict next one future values
    y_expected = targetFunction(x_test)
    y_predicted = model.predict(x_test).reshape(y_expected.shape)

    
    print(y_expected)
    print(y_predicted)
    print(max(abs(np.subtract(y_expected, y_predicted))))

    plt.plot(x_test, y_expected, label="reality")
    plt.plot(x_test, y_predicted, label="prediction")
    plt.legend(loc='upper left')
    plt.ylim(0,1)
    plt.show()
    

