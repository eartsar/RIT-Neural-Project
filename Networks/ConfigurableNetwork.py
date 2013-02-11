from pybrain.tools.validation import Validator
import scipy
from InputCSV import *
from GA import *
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import SigmoidLayer
from numpy import array


class ConfigurableNetwork:
    '''
    Contains a network that can have reconfigurable weights.
    Constructor requires:
    dataset - a training dataset used to find findError
    numHidden - the number of hidden nodes in the network
    '''
    net = None
    data = None

    def __init__(self, dataset, numHidden):
        self.data = dataset
        self.net = buildNetwork(dataset.indim, numHidden, dataset.outdim, bias=True, outclass=SigmoidLayer)
        self.net.sortModules()

    def configureNetwork(self, weights, connections):
        '''
        Changes the weights of the network.
        weights - contains the new weights of the network
        connections - array showing the connectivity of the network. 1 for a connection, 0 for no connection.
        '''
        weights = array(weights)
        connections = array(connections)
        updatedParams = weights * connections
        for i in range(len(self.net.params)):
            self.net.params[i] = updatedParams[i]

    def findMSE(self):
        out = self.net.activateOnDataset(self.data)
        meanSquaredError = Validator.MSE(out, self.data['target'])
        return meanSquaredError

    def findError(self):
        out = self.net.activateOnDataset(self.data)
        out = out.argmax(axis=1)
        errors = 0
        for i in range(len(out)):
            if self.data['class'][i] != out[i]:
                errors += 1
        return errors


def test2():
    numInputs = 8
    numOutputs = 10
    numHidden = 8

    # Import in the yeast Dataset
    csv_importer = CSVImporter()
    dataSet = csv_importer.importFromCSV("../datasets/Yeast/yeast_clean.data", numInputs, numOutputs)
    tstdata, trndata = dataSet.splitWithProportion(0.25)
    tstdata._convertToOneOfMany()
    trndata._convertToOneOfMany()

    # Create a Configurable Network and test out its methods
    configNet = ConfigurableNetwork(trndata, numHidden)

    numEpochs = 50

    popsize = 100
    # this assumes only a single hidden layer
    numConnections = numInputs * numOutputs * numHidden

    Population = makePopulation(popsize, numConnections)

    error = [0] * numEpochs

    for i in range(0, numEpochs):
        for j in range(0, popsize):
            curMember = Population[j]

            configNet.configureNetwork(curMember.Weights, curMember.OnOff)

            f = configNet.findError()

            Population[j].Fitness = f

        sortPop(Population)

        error[i] = Population[0].Fitness
        print "Error at", i, ':', [Population[x].Fitness for x in range(10)]
        Population = breedPop(Population, .1)

        Population = mutate(Population, .1)

    # print("Best weights: \n", Population[0].Weights)
    # print("Best connections: \n", Population[0].OnOff)
    print("Best fitness: \n", Population[0].Fitness)
    print("Error across epochs: \n", error)


def test():
    '''
    Tests out the class and shows how to use it.
    '''
    # Import in the yeast Dataset
    csv_importer = CSVImporter()
    dataSet = csv_importer.importFromCSV("../datasets/Yeast/yeast_clean.data", 8, 10)
    tstdata, trndata = dataSet.splitWithProportion(0.25)
    tstdata._convertToOneOfMany()
    trndata._convertToOneOfMany()

    # Create a Configurable Network and test out its methods
    configNet = ConfigurableNetwork(tstdata, 8)

    # print out the current weights and error
    print "Current Weights:\n", configNet.net.params
    print 'Current Error: ', configNet.findError()

    # Turn off some of the connections
    connections = scipy.ones(len(configNet.net.params))  # Array of ones of the same size as the param array
    connections[0] = 0
    print 'New connections vector:\n', connections

    # Change the network config
    configNet.configureNetwork(configNet.net.params, connections)

    # print out the current weights and error
    print "Updated Weights:\n", configNet.net.params
    print 'Next Error: ', configNet.findError()


if __name__ == '__main__':
    test2()
