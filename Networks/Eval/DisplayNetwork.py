import cPickle
import sys
import matplotlib.pyplot as plt


def graphNetwork(network, showConnected=True, showMissing=False):
    spacing = 5
    fig = plt.figure(0)
    fig.suptitle('Network Connections')
    inputNodes = []
    hiddenNodes = []
    outputNodes = []
    biasNodes = []
    for i in range(network.indim):
        inputNodes.append((0, i * spacing))
    for i in range(network['hidden0'].dim):
        hiddenNodes.append((5, i * spacing))
    for i in range(network.outdim):
        outputNodes.append((10, i * spacing))
    for i in range(network['bias'].dim):
        biasNodes.append((0, -5 - i))

    inToHid = network.connections[network['in']][0]
    for weightArg in range(len(inToHid.params)):
        nodes = inToHid.whichBuffers(weightArg)
        if(inToHid.params[weightArg] != 0):
            if(showConnected):
                plt.plot((inputNodes[nodes[0]][0], hiddenNodes[nodes[1]][0]), (inputNodes[nodes[0]][1],
                         hiddenNodes[nodes[1]][1]), 'k-')
        else:
            if(showMissing):
                plt.plot((inputNodes[nodes[0]][0], hiddenNodes[nodes[1]][0]), (inputNodes[nodes[0]][1],
                         hiddenNodes[nodes[1]][1]), 'r-')

    hidToOut = network.connections[network['hidden0']][0]
    for weightArg in range(len(hidToOut.params)):
        nodes = hidToOut.whichBuffers(weightArg)
        if(hidToOut.params[weightArg] != 0):
            if(showConnected):
                plt.plot((hiddenNodes[nodes[0]][0], outputNodes[nodes[1]][0]), (hiddenNodes[nodes[0]][1],
                         outputNodes[nodes[1]][1]), 'k-')
        else:
            if(showMissing):
                plt.plot((hiddenNodes[nodes[0]][0], outputNodes[nodes[1]][0]), (hiddenNodes[nodes[0]][1],
                         outputNodes[nodes[1]][1]), 'r-')

    biasToHid = network.connections[network['bias']][1]
    for weightArg in range(len(biasToHid.params)):
        nodes = biasToHid.whichBuffers(weightArg)
        if(biasToHid.params[weightArg] != 0):
            if(showConnected):
                plt.plot((biasNodes[nodes[0]][0], hiddenNodes[nodes[1]][0]), (biasNodes[nodes[0]][1],
                         hiddenNodes[nodes[1]][1]), 'k-')
        else:
            if(showMissing):
                plt.plot((biasNodes[nodes[0]][0], hiddenNodes[nodes[1]][0]), (biasNodes[nodes[0]][1],
                         hiddenNodes[nodes[1]][1]), 'r-')

    biasToOut = network.connections[network['bias']][0]
    for weightArg in range(len(biasToOut.params)):
        nodes = biasToOut.whichBuffers(weightArg)
        if(biasToOut.params[weightArg] != 0):
            if(showConnected):
                plt.plot((biasNodes[nodes[0]][0], outputNodes[nodes[1]][0]), (biasNodes[nodes[0]][1],
                         outputNodes[nodes[1]][1]), 'k-')
        else:
            if(showMissing):
                plt.plot((biasNodes[nodes[0]][0], outputNodes[nodes[1]][0]), (biasNodes[nodes[0]][1],
                         outputNodes[nodes[1]][1]), 'r-')

    for i in range(len(inputNodes)):
        plt.plot([inputNodes[i][0]], [inputNodes[i][1]], 'ob')
    for i in range(len(hiddenNodes)):
        plt.plot([hiddenNodes[i][0]], [hiddenNodes[i][1]], 'ob')
    for i in range(len(outputNodes)):
        plt.plot([outputNodes[i][0]], [outputNodes[i][1]], 'ob')
    for i in range(len(biasNodes)):
        plt.plot([biasNodes[i][0]], [biasNodes[i][1]], 'go')
    plt.xlim([-1, 11])
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.show()


def main():
    if(len(sys.argv) != 2):
        print 'Usage: python DisplayNetwork.py [Network.pkl]'
        return 1

    networkFile = open(sys.argv[1])
    network = cPickle.load(networkFile)
    networkFile.close()
    graphNetwork(network, showMissing=True, showConnected=True)

if __name__ == '__main__':
    main()
