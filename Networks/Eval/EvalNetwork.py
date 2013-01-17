import cPickle
import sys
import matplotlib.pyplot as plt
from pybrain.tools.validation import Validator

def plotTrainingError(data):
	plt.figure(0)
	plt.plot(data['errors'])
	plt.show()

def findClassificationRates(network,data):
	testSet = data['test']
	trainSet = data['train']

	#Find the classification rate on the Test Set
	out = network.activateOnDataset(testSet)
	#Find MSE on Test Set
	mse = Validator.MSE(out,testSet['target'])
	print 'MSE(TEST):',mse
	out = out.argmax(axis = 1) 
	correct = 0
	for i in range(len(out)):
		if testSet['class'][i] == out[i]:
			correct += 1
	crTest = float(correct)/len(out)

	#Find the classification rate on the Training Set
	out = network.activateOnDataset(trainSet)
	print out
	print trainSet['target']
	org = out
	mse = Validator.ESS(out,trainSet['target'])
	print 'MSE(Train):',mse, data['errors'][-1]
	out = out.argmax(axis = 1) 
	correct = 0
	for i in range(len(out)):
		mse = Validator.MSE(org[i],trainSet['target'][i])
		#print 'MSE ',i,':',mse
		if trainSet['class'][i] == out[i]:
			correct += 1
	crTrain = float(correct)/len(out)
	print "Classification Rate(Test):", crTest
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
	#plotTrainingError(data)
	findClassificationRates(network,data)

if __name__ == '__main__':
	main()