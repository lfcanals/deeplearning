import numpy as np
import matplotlib.pyplot as plt

import LSTM
import logdiscBase

R = 3.92
x0 = 0.1234
 
#
# x it should be a continuous monotone increasing sequence  or a single number
#
def logdisc(x):
    return logdiscBase.logdiscBase(R, x0, x)

LSTM.run(logdisc, logdiscBase.x_train, logdiscBase.x_tests)
