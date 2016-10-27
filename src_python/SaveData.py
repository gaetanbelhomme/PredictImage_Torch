import numpy as np
from matplotlib import pyplot
import nrrd
import argparse
import os
from six.moves import cPickle as pickle


# PARSER :

parser = argparse.ArgumentParser()

parser.add_argument('-i', action='store', dest='inputFolder', help='folder with images we want to convert into nrrd',
                    default='../Data/outputNpy/')
parser.add_argument('-o', action='store', dest='outputFolder', help='save results in this folder',
                    default='../Data/outputNrrd/')
parser.add_argument('-header', action='store', dest='headers', help='headers.pickle',
                    default='../Data/input/DataNrrd2D_headers.pickle')
parser.add_argument('-pos', action='store', dest='posInit', help='indice of the first image input folder',
                    default='0', type=int)

options = parser.parse_args()

# Initialization :

input = options.inputFolder
output = options.outputFolder
headers = options.headers
posInit = options.posInit

# Import headers :


with open(headers, 'rb') as f:
    save = pickle.load(f)
    train_header_start = save['train_header_start']
    train_header_end = save['train_header_end']
    valid_header_start = save['valid_header_start']
    valid_header_end = save['valid_header_end']
    test_header_start = save['test_header_start']
    test_header_end = save['test_header_end']
    del save  # hint to help gc free up memory

print ""
print("Training headers", np.shape(train_header_start), np.shape(train_header_end))
print('Validation headers', np.shape(valid_header_start), np.shape(valid_header_end))
print('Test headers', np.shape(test_header_start), np.shape(test_header_end))


if os.path.isdir(input):
    print input, "is present"
else:
    raise Exception('Failed to find %s' % input)

if os.path.isdir(output):
    print output, "is present"
else:
    raise Exception('Failed to find %s' % output)

image_files = os.listdir(input)

for file in image_files:
    filename, ext = os.path.splitext(file)

    if ext == '.npy':
        image = np.load(input + file)
        nrrd.write(output + filename + '.nrrd', image, test_header_start[posInit])


