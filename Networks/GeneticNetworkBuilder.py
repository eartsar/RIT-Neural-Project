import sys
from InputCSV import *
from ConfigurableNetwork import *
from time import clock
import sys
import cPickle

def buildAndTrainNetwork(trndata, numHidden, numEpochs=100, popsize =100):

      #Create a Configurable Network
      configNet = ConfigurableNetwork(trndata,numHidden)
      numInputs = trndata.indim
      numOutputs = trndata.outdim

      #this assumes only a single hidden layer
      numConnections = numInputs * numOutputs * numHidden
      
      Population = makePopulation(popsize,numConnections)
      error = [0] * numEpochs

      #Start the clock
      start = clock()

      #Initialize network fitness
      for j in range(0, popsize):
          curMember = Population[j]
          configNet.configureNetwork(curMember.Weights, curMember.OnOff)
          f = configNet.findError()
          Population[j].Fitness = f
      sortPop(Population)
      error[0] = Population[0].Fitness
      print "Top 5 error at 0:",[Population[x].Fitness for x in range(5)]

      for i in range(1, numEpochs):

            Population = breedPop(Population, .1)
            Population = mutate(Population, .1)

            for j in range(0, popsize):
                  curMember = Population[j]

                  configNet.configureNetwork(curMember.Weights, curMember.OnOff)

                  f = configNet.findError()

                  Population[j].Fitness = f

            sortPop(Population)
            configNet.configureNetwork(Population[0].Weights, Population[0].OnOff)
            error[i] = configNet.findMSE()
            print "Lowest 5 error count at",i,':',[Population[x].Fitness for x in range(5)] 
            
      configNet.configureNetwork(Population[0].Weights, Population[0].OnOff)

      #Stop the clock
      end = clock()
      elapsed = end - start
      print 'Start, End:',start,':',end
      print 'Elapsed:',elapsed

      print("Best fitness: \n", Population[0].Fitness)
      return configNet.net,error

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
        print 'Usage: python GeneticNetworkBuilder.py <input_filename> <network_name> <num_attributes> <num_classifications> <num_hidden_nodes>'
        return 1

    input_filename = sys.argv[1]
    network_name = sys.argv[2]
    num_attributes = int(sys.argv[3])
    num_classifications = int(sys.argv[4])
    num_hidden = int(sys.argv[5])
    csv_importer = CSVImporter()
    dataSet = csv_importer.importFromCSV(input_filename, num_attributes, num_classifications)
    tstdata, trndata = dataSet.splitWithProportion(0.25)
    tstdata._convertToOneOfMany()
    trndata._convertToOneOfMany()
    network,epochErrors = buildAndTrainNetwork(trndata,num_hidden,numEpochs=50, popsize =100)
    saveNetworkAndData(network_name, network, trndata, tstdata, epochErrors)

if __name__ == '__main__':
    main()
