import keras
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import RNN 


# Transpose function
###### MAIN CODE 

def run(targetFunction):
    lengthForecast = 1000

    # Two layers, dense
    neuronsPerLayer = 128
    numOfLayers = 1
    
    model = Sequential()
    for i in range(numOfLayers-1):
        model.add(LSTM(neuronsPerLayer, return_sequences=True, activation='tanh'))

    model.add(LSTM(neuronsPerLayer, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='mse', optimizer='SGD')
    
    
    
    # Train it
    x_train = np.arange(0, 1000, dtype=float)
    y_train = targetFunction(np.add(lengthForecast, x_train))
    
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    #X = X.reshape((X.shape[0], X.shape[1], 1))  # Only 1 feature
    X = np.asarray(x_train).reshape(1000,1,1)
    y = np.asarray(y_train)

    print(X)
    print(y)
    
    history = model.fit(X, y, epochs=25)
    
    
    # Predict next one future values
    x_test = np.arange(100, 120, dtype=float)
    y_expected = targetFunction(np.add(lengthForecast, x_test))
    
    y_test = []
    for i in range(len(x_test)):
        input_test = x_test[i].reshape(1, 1, 1)
        y_test = np.append(y_test, model.predict(input_test))

    
    plt.plot(x_test, y_test, label="prediction")
    plt.plot(x_test, y_expected, label="reality")
    plt.legend(loc='upper left')
    plt.ylim(-2,2)
    plt.show()
    
    print(max(abs(np.subtract(y_expected, y_test))))

