import numpy as np

import Dense

def targetF(x):
    # For values between 0 and 1
    return (np.sin(2*np.pi*x)+1)/2

x_train = np.arange(0, 0.99, 0.001, dtype=float)
x_test = np.arange(0.99, 1, 0.001, dtype=float)
Dense.run(targetF, x_train, x_test)