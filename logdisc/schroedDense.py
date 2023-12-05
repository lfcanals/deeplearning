import numpy as np
import Dense

y0 = 0.2
r = 4

def schroedingerLogisticSolution(x):
    # x between 0 and 1 in steps of .001
    x = x * 1000
    xPrev = x[0] - 1
    yPrev = y0
    for i in range(0,int(x[0])):
        yPrev = r * yPrev * (1-yPrev)

    y = []
    for i in x:
        i = np.floor(i)
        if(i != xPrev+1):
            raise "Vector of input should be sequential"

        yPrev = r * yPrev * (1-yPrev)
        y.append(yPrev)
        xPrev = i
    return np.asarray(y)


trainX = np.arange(0, 0.9, .001, dtype=float)
testX = np.arange(0.9, 1, .001, dtype=float)
Dense.run(schroedingerLogisticSolution, trainX, testX)
