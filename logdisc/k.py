import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import TimeseriesGenerator

X_train = np.arange(0,2*np.pi,0.01)
y_train = np.sin(X_train)

X_test = np.arange(0.5,6.5,0.01) 
y_test = np.sin(X_test)

n_features = 1

train_series = y_train.reshape((len(y_train), n_features))
test_series  = y_test.reshape((len(y_test), n_features))

print(train_series.shape)

look_back  = 99

train_generator = TimeseriesGenerator(train_series, train_series,
                                      length        = look_back,
                                      sampling_rate = 1,
                                      stride        = 1,
                                      batch_size    = 1)

test_generator = TimeseriesGenerator(test_series, test_series,
                                      length        = look_back,
                                      sampling_rate = 1,
                                      stride        = 1,
                                      batch_size    = 1)
print('len(train_generator)=', len(train_generator))
print('len(train_generator[0])=', len(train_generator[0]))
print('len(train_generator[0][0])=', len(train_generator[0][0]))
print('len(train_generator[0][0][0])=', len(train_generator[0][0][0]))
print(train_generator[0][0]);

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

n_neurons  = 4
model = Sequential()
model.add(LSTM(n_neurons, input_shape=(look_back, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(train_generator,epochs=300, verbose=0)

extrapolation = list()
seed_batch    = y_test[:look_back].reshape((1,look_back, n_features))
current_batch = seed_batch

# extrapolate the next 180 values
for i in range(180):
    predicted_value = model.predict(current_batch)[0]
    extrapolation.append(predicted_value)
    current_batch = np.append(current_batch[:,1:,:],[[predicted_value]],axis=1)

plt.plot(extrapolation)
plt.show()
