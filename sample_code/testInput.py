from InputCSV import *
import sys


def main():
    if len(sys.argv) != 4:
        print 'Usage: python testInput.py <input_filename> <num_attributes> <num_classifications>'
        return 1

    input_filename = sys.argv[1]
    num_attributes = int(sys.argv[2])
    num_classifications = int(sys.argv[3])
    csv_importer = CSVImporter()
    dataSet = csv_importer.importFromCSV(input_filename, num_attributes, num_classifications)
    for inpt, target in dataSet:
        print inpt, target

if __name__ == '__main__':
    main()
