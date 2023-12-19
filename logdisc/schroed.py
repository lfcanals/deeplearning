import numpy as np

import LSTM
import logdiscBase


x0 = 0.1234
 
#
# x it should be a continuous monotone increasing sequence  or a single number
#
def schored(x):
    return logdiscBase.logdiscBase(4, x0, x)

LSTM.run(schored, logdiscBase.x_train, logdiscBase.x_tests)
