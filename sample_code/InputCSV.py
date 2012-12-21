from pybrain.datasets import SupervisedDataSet


class CSVImporter:
    def importFromCSV(self, fileName, numInputs, numOutputs):
        """
        Function that reads in a CSV file and passes on to the pybrain
        neural net dataset structure to be used with the library's
        neural net classes.

        It expects that the last columns (determined by numOutputs) to be
        the classification columns.
        """
        dataSet = SupervisedDataSet(numInputs, numOutputs)
        dataFile = open(fileName)
        for line in dataFile:
            data = [float(x) for x in line.strip().split(',') if x != '']
            inputData = tuple(data[:numInputs])
            outputData = tuple(data[numInputs:])
            dataSet.addSample(inputData, outputData)

        dataFile.close()
        return dataSet
