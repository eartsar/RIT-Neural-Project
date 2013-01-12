import sys
import math

"""
This collection of utility function is meant to clean and fix the abalone dataset.
Author: Eitan Romanoff
"""


def load_file(filename):
    """
    Function that loads in the yeast file, and converts it to CSV format.
    Furthermore, we cut out the first attribute because it's unique.
    """
    lines = []
    in_file = open(filename)
    for line in in_file:
        line = line.strip()
        csvline = ','.join(line.split()[1:])
        lines.append(csvline)
    in_file.close()
    return lines


def write_file(entries, filename):
    """
    Function that writes out the fixed abalone CSV file
    """
    out_file = open(filename, 'w')
    for entry in entries:
        out_file.write(entry + '\n')
    out_file.close()


def move_classifier(entries):
    """
    Function that moves the classifier from the first column to the last column
    """
    for line_num in range(len(entries)):
        entry = entries[line_num]
        pre = entry[:1]
        entry = entry[2:] + ',' + pre
        entries[line_num] = entry
    return entries


def normalize_fields(entries):
    """
    Function that centers and noramlizes (divide by stdev) the normalize_fields
    """
    # first we center
    num_attributes = len(entries[0].split(','))
    # subtract 2 for classifier exclusion
    means = []
    stdevs = []
    for attribute_index in range(num_attributes - 1):
        mean = get_mean(entries, attribute_index)
        stdev = get_stdev(entries, attribute_index, mean)
        # print 'Attribute ' + str(attribute_index) + ' - mean: ' + str(mean) + '  stdev: ' + str(stdev)
        means.append(mean)
        stdevs.append(stdev)

    for line_num in range(len(entries)):
        entry = entries[line_num]
        unpacked = entry.split(',')

        # exclude the classifier
        entry_rebuild = ''
        for i in range(len(unpacked) - 1):
            unpacked[i] = float(unpacked[i]) - means[i]
            unpacked[i] = unpacked[i] / stdevs[i]
            entry_rebuild = entry_rebuild + str(unpacked[i]) + ','

        entries[line_num] = entry_rebuild + unpacked[-1]

    return entries


def get_stdev(entries, attribute_index, mean):
    """
    Function that gets the stdev
    """
    # assume CSV with no spaces
    vals = []
    for entry in entries:
        val = float(entry.split(',')[attribute_index])
        val = val - mean
        val = val * val
        vals.append(val)
    stdev_sq = sum(vals) / len(vals)
    stdev = math.sqrt(stdev_sq)
    return stdev


def get_mean(entries, attribute_index):
    """
    Function that gets the mean
    """
    vals = []
    for entry in entries:
        items = entry.split(',')
        vals.append(float(items[attribute_index]))
    mean = sum(vals) / len(vals)
    return mean


def convert_classes(entries):
    """
    Function that converts classes to numeric class labels
    and addes meta data for class names
    """
    attrib_list = []
    classLabel = []
    ret_list = []

    for entry in entries:
        items = entry.split(',')
        attrib_list = items[:-1]

        classification = items[-1]
        if classification == 'CYT':
            classLabel = '0'
        elif classification == 'NUC':
            classLabel = '1'
        elif classification == 'MIT':
            classLabel = '2'
        elif classification == 'ME3':
            classLabel = '3'
        elif classification == 'ME2':
            classLabel = '4'
        elif classification == 'ME1':
            classLabel = '5'
        elif classification == 'EXC':
            classLabel = '6'
        elif classification == 'VAC':
            classLabel = '7'
        elif classification == 'POX':
            classLabel = '8'
        elif classification == 'ERL':
            classLabel = '9'

        full_list = attrib_list[:] + [classLabel]
        entry_rebuild = ','.join(full_list)
        ret_list.append(entry_rebuild)
    ret_list.insert(0,'!labels:,CYT,NUC,MIT,ME3,ME2,ME1,EXC,VAC,POX,ERL')
    return ret_list


def main():
    if len(sys.argv) != 3:
        print 'Usage: python abafix.py <input_filename> <output_filename'
        return 1

    # Load the file
    entries = load_file(sys.argv[1])
    # normalize the non-classifier column
    entries = normalize_fields(entries)
    # split the classifier into multiple columns for neural net output vector
    entries = convert_classes(entries)
    # write the fixed data
    write_file(entries, sys.argv[2])


if __name__ == '__main__':
    main()
