import numpy as np

import LSTM


 
#
# x it should be a continuous monotone increasing sequence  or a single number
#
def logdiscBase(R, x0, x):
    def calculateFromBeginning(initVal, constant, iteration):
        z = initVal
        for a in range(1,iteration): z = constant * z * (1-z)
        return z


    if type(x) is int:
        return calculateFromBeginning(x0, R, x)

    else:
        i = x[0]
        z = calculateFromBeginning(x0, R, i)
        yy = [z]
        previousI = x[0]
        for i in x[1:]:
            if(i!=previousI+1):
                raise Exception('Input sequence for logdisc function should ' \
                                'be a consecutive list of intgers')
            previousI = i
            z = R * z * (1-z)
            yy.append(z)

        return np.asarray(yy)
