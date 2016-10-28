import ConfigData as CD
import numpy as np
import os
from six.moves import cPickle as pickle
import nrrd
import sys

# INITIALIZATION :
test = CD.test
train = CD.train
start = CD.start
end = CD.end

test_size = CD.test_size
train_size = CD.train_size
valid_size = CD.valid_size


# MAIN :

# Check folders :
test = CD.CheckFolders(test)
train = CD.CheckFolders(train)

test_start = CD.CheckFolders(folder=start, root_name=test)
test_end = CD.CheckFolders(folder=end, root_name=test)

train_start = CD.CheckFolders(folder=start, root_name=train)
train_end = CD.CheckFolders(folder=end, root_name=train)


# Create a pickle per folder
train_datasets, train_header = CD.maybe_pickle(train_start, train_end)
test_datasets, test_header = CD.maybe_pickle(test_start, test_end)


# Given sizes create valid, training and testing dataset
valid_dataset_start, valid_dataset_end, valid_set_headers, train_dataset_start, train_dataset_end, train_set_headers \
    = CD.merge_datasets(train_datasets, train_header, train_size, valid_size)
_, _, _, test_dataset_start, test_dataset_end, test_set_headers = CD.merge_datasets(test_datasets, test_header, test_size)

print('Training:', train_dataset_start.shape, train_dataset_end.shape)
print('Validation:', valid_dataset_start.shape, valid_dataset_end.shape)
print('Testing:', test_dataset_start.shape, test_dataset_end.shape)

print('Headers:')
print('training', np.shape(train_set_headers))
print('validation', np.shape(valid_set_headers))
print('testing', np.shape(test_set_headers))

# Randomize datasets
train_dataset_start, train_dataset_end, train_set_headers = CD.randomize(train_dataset_start, train_dataset_end,
                                                                         train_set_headers)
test_dataset_start, test_dataset_end, test_set_headers = CD.randomize(test_dataset_start, test_dataset_end,
                                                                      test_set_headers)
valid_dataset_start, valid_dataset_end, valid_set_headers = CD.randomize(valid_dataset_start, valid_dataset_end,
                                                                         valid_set_headers)


# Resize images : 64*64 -> resize=true : 1*64*64 / resize=false 64*64*1
train_dataset_start, train_dataset_end = CD.reformat(train_dataset_start, train_dataset_end)
valid_dataset_start, valid_dataset_end = CD.reformat(valid_dataset_start, valid_dataset_end)
test_dataset_start, test_dataset_end = CD.reformat(test_dataset_start, test_dataset_end)

print('Training set', train_dataset_start.shape, train_dataset_end.shape)
print('Validation set', valid_dataset_start.shape, valid_dataset_end.shape)
print('Test set', test_dataset_start.shape, test_dataset_end.shape)

# Save Data [.npz, .npy or .pickle ]
save = {
        'train_dataset_start': train_dataset_start,
        'train_dataset_end': train_dataset_end,
        'valid_dataset_start': valid_dataset_start,
        'valid_dataset_end': valid_dataset_end,
        'test_dataset_start': test_dataset_start,
        'test_dataset_end': test_dataset_end,
    }

save_header = {
        'train_header_start' : train_set_headers[0],
        'train_header_end' : train_set_headers[1],
        'test_header_start': test_set_headers[0],
        'test_header_end': test_set_headers[1],
        'valid_header_start': valid_set_headers[0],
        'valid_header_end': valid_set_headers[1],
}

CD.store(save)
CD.store(save_header, type='pickle', name=CD.headeroutput)

#print('start :', test_set_headers[0])
