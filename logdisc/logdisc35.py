import numpy as np

import LSTM
import logdiscBase
R = 3.5
x0 = 0.1234
 
#
# x it should be a continuous monotone increasing sequence  or a single number
#
def logdisc(x):
    return logdisc.logdiscBase(R, x0, x)

x_train = []
rng = np.arange(0, 500)
rngVariations = np.arange(0, 100)

for i in rng: x_train.append(rngVariations + i)

x_tests = []
for x in np.arange(500, 600, 1, dtype=int):
    x_tests.append(np.arange(x, x+99.99, 1, dtype=int))

LSTM.run(logdisc, x_train, x_tests)
