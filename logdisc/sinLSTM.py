import numpy as np

import LSTM

def targetF(x):
    # For values between 0 and 1
    return (np.sin(2*np.pi*x)+1)/2

x_train = []
rng = np.arange(0, 1, 0.01)
rngVariations = np.arange(0, 0.5, 0.001)

for i in rng: x_train.append(rngVariations + i)

x_tests = []

for x in np.arange(0.50, 1.3, 0.01, dtype=float):
    x_tests.append(np.arange(x, x+0.999, 0.01, dtype=float))

print(x_tests)
LSTM.run(targetF, x_train, x_tests)
