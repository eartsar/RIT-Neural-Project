import cPickle
import sys
import matplotlib.pyplot as plt

def plotTrainingError(data):
	fig = plt.figure(0)
	fig.suptitle('Mean Squared Error vs Number of Epochs')
	plt.xlabel('Number of Epochs')
	plt.ylabel('Mean Squared Error')
	plt.plot(data['errors'][:-1])
	plt.show()

def findClassificationRates(network,data):
	testSet = data['test']
	trainSet = data['train']

	#Find the classification rate on the Test Set
	out = network.activateOnDataset(testSet)
	out = out.argmax(axis = 1) 
	correct = 0
	for i in range(len(out)):
		if testSet['class'][i] == out[i]:
			correct += 1
	crTest = float(correct)/len(out)
	numCorrectTest = correct
	numIncorrectTest = len(out)-correct

	#Find the classification rate on the Training Set
	out = network.activateOnDataset(trainSet)
	out = out.argmax(axis = 1) 
	correct = 0
	for i in range(len(out)):
		if trainSet['class'][i] == out[i]:
			correct += 1
	crTrain = float(correct)/len(out)
	print "Correct Classification - Incorrect Classification(Test):",numCorrectTest,numIncorrectTest 
	print "Classification Rate(Test):", crTest
	print "Correct Classification - Incorrect Classification(Train):",correct,(len(out)-correct) 
	print "Classification Rate(Train):", crTrain

def main():
	if(len(sys.argv) !=  3):
		print 'Usage: python EvalNetwork.py [Network.pkl] [NetworkData.pkl]'
		return 1

	networkFile = open(sys.argv[1])
	networkDataFile = open(sys.argv[2])
	network = cPickle.load(networkFile)
	data = cPickle.load(networkDataFile)
	networkFile.close()
	networkDataFile.close()
	plotTrainingError(data)
	findClassificationRates(network,data)

if __name__ == '__main__':
	main()