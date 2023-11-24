import keras
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


# Transpose function
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)



###### MAIN CODE 


n_steps = 20


# Two layers, dense
model = Sequential()
model.add(LSTM(100, activation='tanh'))
model.add(Dense(1, activation='tanh'))

model.compile(loss='mse', optimizer='adam')



# Train it
x_train = np.arange(-50*np.pi, 50*np.pi, 0.1)
y_train = np.sin(x_train)

X, y = split_sequence(y_train, n_steps)
# reshape from [samples, timesteps] into [samples, timesteps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))  # Only 1 feature

history = model.fit(X, y, epochs=10)


# Predict some future values
x_test = np.arange(52*np.pi, 62*np.pi, 0.1)
y_expected = np.sin(x_test)
y_test = y_expected[:n_steps]

for i in range(len(x_test) - n_steps):
    input_test = y_test[i : i + n_steps].reshape((1, n_steps, 1)) # Only 1 feature
    y_pred = model.predict(input_test)
    y_test = np.append(y_test, y_pred)


plt.plot(x_test[n_steps:], y_test[n_steps:], label="prediction")
plt.plot(x_test, y_expected, label="reality")
plt.legend(loc='upper left')
plt.ylim(-2,2)
plt.show()

print(max(abs(np.subtract(y_pred, y_test))))
