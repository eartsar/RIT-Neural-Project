from InputCSV import *
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, TanhLayer, BiasUnit
from pybrain.structure import FullConnection
from time import clock
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
import sys
import cPickle

def standardBuildMLP(dataSet, num_hidden):
    net = buildNetwork(dataSet.indim, num_hidden, dataSet.outdim, bias=True, outclass=SigmoidLayer)
    return net

def buildMLP(dataSet, num_hidden):
    '''
    Function that builds a feed forward network based
    on the datset inputed.
    The hidden layer has nodes equal to num_hidden.
    '''
    #make the network
    network = FeedForwardNetwork()
    #make network layers
    inputLayer = LinearLayer(dataSet.indim)
    hiddenLayer = SigmoidLayer(num_hidden)
    outputLayer = LinearLayer(dataSet.outdim)

    #add the layers to the network
    network.addInputModule(inputLayer)
    network.addModule(hiddenLayer)
    network.addOutputModule(outputLayer)

    #add bias
    network.addModule(BiasUnit(name='bias'))

    #create connections between layers
    inToHidden = FullConnection(inputLayer, hiddenLayer)
    hiddenToOut = FullConnection(hiddenLayer, outputLayer)

    #connect bias
    network.addConnection(FullConnection(network['bias'], outputLayer))
    network.addConnection(FullConnection(network['bias'], hiddenLayer))

    #add connections to the network
    network.addConnection(inToHidden)
    network.addConnection(hiddenToOut)

    network.sortModules()
    return network


def findSumSquredError(output, target):
    error = 0
    for i in range(len(output)):
        for j in range(len(output[i])):
            error = error + ((output[i][j] - target[i][j]) ** 2)
    return error


def trainNetwork(network, trainData, maxEpochs=None, verbose=False):
    '''
    Trains the inputed network on the training data until convergence.
    Optionaly maxEpochs can be set to put an uperbound on the number of epochs.
    Returns an Array of the training error at each epoch.
    '''
    trainer = BackpropTrainer(network, dataset=trainData, verbose=verbose, learningrate=0.1)
    trainErrors = []
    start = clock()
    trainErrors,valErrors = trainer.trainUntilConvergence(maxEpochs = maxEpochs, verbose = verbose)
    end = clock()
    elapsed = end - start
    print 'Start, End:',start,':',end
    print 'Elapsed:',elapsed
    '''
    sse = 1
    i = 1
    while sse > 0.01 and trainer.totalepochs <= maxEpochs:
        trainer.train()
        output = network.activateOnDataset(trainData)
        sse = findSumSquredError(output, trainData['target'])
        trainErrors.append(sse)
        print 'SSE(', i, '): ', sse
        i += 1
    '''
    return trainErrors


def saveNetworkAndData(networkName, network, trainData=None, testData=None, epochErrors=None):
    '''
    Creates two pickle files. One for the network and the other for data related to the network.
    The file networkname+'_network.pkl' stores the network.
    The file networkname+'_data.pkl' stores the data.
    Data consists of the training data, testing data, and error at each epoch stored into a dictionary
        with the keywords  'train', 'test', and 'errors' respectively
    '''
    pickleFile = open(networkName + "_network.pkl", 'wb')
    cPickle.dump(network, pickleFile)
    pickleFile.close()
    networkData = {'train': trainData, 'test': testData, 'errors': epochErrors}
    pickleFile = open(networkName + "_data.pkl", 'wb')
    cPickle.dump(networkData, pickleFile)
    pickleFile.close()


def main():
    if len(sys.argv) != 6:
        print 'Usage: python MLPBuilder.py <input_filename> <network_name> <num_attributes> <num_classifications> <num_hidden_nodes>'
        return 1

    input_filename = sys.argv[1]
    network_name = sys.argv[2]
    num_attributes = int(sys.argv[3])
    num_classifications = int(sys.argv[4])
    num_hidden = int(sys.argv[5])
    csv_importer = CSVImporter()
    dataSet = csv_importer.importFromCSV(
        input_filename, num_attributes, num_classifications)
    tstdata, trndata = dataSet.splitWithProportion(0.25)
    tstdata._convertToOneOfMany()
    trndata._convertToOneOfMany()
    network = standardBuildMLP(trndata, num_hidden)
    epochErrors = trainNetwork(
        network, trndata, maxEpochs=1000, verbose=True)
    saveNetworkAndData(network_name, network, trndata, tstdata, epochErrors)

if __name__ == '__main__':
    main()
