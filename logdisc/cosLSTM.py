import numpy as np

import LSTM

def targetF(x):
    return (np.cos(x)+1)/2

x_train = []
rng = np.arange(0, 50, 0.1)
rngVariations = np.arange(0, 10, 0.1)

for i in rng: x_train.append(rngVariations + i)

x_tests = []
for x in np.arange(50, 60, 0.1, dtype=float):
    x_tests.append(np.arange(x, x+9.999, 0.1, dtype=float))

LSTM.run(targetF, x_train, x_tests)
