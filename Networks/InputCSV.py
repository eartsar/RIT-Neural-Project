from pybrain.datasets import ClassificationDataSet

class Whatever:
    pass

class CSVImporter:
    def importFromCSV(self, fileName, numInputs, numClasses):
        """
        Function that reads in a CSV file and passes on to the pybrain
        neural net dataset structure to be used with the library's
        neural net classes.

        It expects that the last columns (determined by numOutputs) to be
        the classification columns.
        """
        dataSet = None
        dataFile = open(fileName)
        line = dataFile.readline()
        data = [str(x) for x in line.strip().split(',') if x != '']
        if(data[0] == '!labels:'):
            labels = data[1:]
            dataSet = ClassificationDataSet(numInputs, nb_classes=numClasses, class_labels=labels)
            line = dataFile.readline()
        else:
            dataSet = ClassificationDataSet(numInputs, nb_classes=numClasses)

        while line != '':
            data = [float(x) for x in line.strip().split(',') if x != '']
            inputData = data[:numInputs]
            outputData = data[-1:]
            dataSet.addSample(inputData, outputData)
            line = dataFile.readline()

        dataFile.close()
        return dataSet
