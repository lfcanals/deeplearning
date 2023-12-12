import numpy as np

def checkArrayDimensions(rnnFirstLayerType, trainInCases, trainOutCases):
    match rnnFirsLayerType:
        case 'LSTM':
            if(trainInCases.shape[0] != trainOutCases.shape[0]):
                print('Input train cases should be an array of N elements')
                print('The output cases should be an array of N elemenets also')
                print('Each of the elements of the input array should be another array\
                        of M elements')
                print('So, input test Cases shape should be (N,M)')
                print('Output cases should be (N,x)')
                print('The value of x depends on the last layer of the network')
                print('Sometimes you want just a single category or value,\
                       but sometimes you need a vector. It is up to you')
                return

            # Verify that all arrays in the input are of the same size

class DataSet:
    """To arrays: one of inputs, one of outputs with the train examples"""
    def __init__(self):
        self.trainInput = list() 
        self.trainOutput = list()
        self.trainBuilt = False

        self.testInput = list() 
        self.testBuilt = False
        

        

class LSTMDataSet(DataSet):
    def addTrainExample(self, trainInputs, trainOutputs):
        """Add a pair of train example trainInput and example trainOutput"""

        self.trainBuilt = False
        if(len(self.trainInput) > 0): assert len(trainInputs) == len(self.trainInput[0])

        self.trainInput.append(trainInputs)
        self.trainOutput.append(trainOutputs)


    def addTestExample(self, x):
        """Add a test example testInput"""

        self.testBuilt = False
        if(len(self.testInput) > 0): 
            if(len(x) != self.testInput[0].shape[1]):
                print('ERROR: the first test input has ', self.testInput[0].shape[1], \
                      ' elements, but the supplied test input has ', len(x), \
                      'Sorry, but they should be the same size')
                raise Exception('ERROR In test set dimensions')

        self.testInput.append(x.reshape(1,x.shape[0],1))


    def build(self):
        """Prepares vectors and arrays to be trained"""
        if(self.trainBuilt==False):
            self.trainBuilt = True
            if(len(self.trainInput) > 0):
                self.trainBuiltInput = np.array(self.trainInput)
                self.trainBuiltOutput = np.array(self.trainOutput)
                self.trainBuiltInput = self.trainBuiltInput.reshape(\
                        (self.trainBuiltInput.shape[0], self.trainBuiltInput.shape[1],1))


    def getDataSets(self):
        """Return input and output vectors ready to be used for fitting"""
        if(self.trainBuilt==False): self.build()
        return self.trainBuiltInput, self.trainBuiltOutput


    def getTestSet(self):
        """Return input vectors ready to be used for testing"""
        return self.testInput


    def summary(self):
        print('Summary of test and training data')
        print('---------------------------------')
        print('Number of train cases:', len(self.trainInput))
        if(len(self.trainInput)>0):
            print('Size of each train case:', len(self.trainInput[0]))

        print('Number of test cases:', len(self.testInput))
        if(len(self.testInput)>0):
            print('Size of each test case:', self.testInput[0].shape[1])
