import numpy as np
import RNN

x0 = 0.2
l = np.power(np.sin(np.power(2,x0)),2)

def schroedingerLogisticSolution(x):
    return np.power(np.sin(l * np.power(2,x)), 2)

RNN.run(schroedingerLogisticSolution)
