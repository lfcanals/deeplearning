# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.

# For Python 2.7, install Pillow not PIL

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import ndimage
from six.moves import cPickle as pickle

import hashlib

pickle_file = 'notMNIST.pickle'



f = open(pickle_file, 'r')
data = pickle.load(f)

trainHashMap = {}
for i,image in enumerate(data['train_dataset']):
  trainHashMap[hashlib.md5(image)] = i
print('Train set size ' + str(len(trainHashMap)))
  
validHashMap = {}
for i,image in enumerate(data['valid_dataset']):
  validHashMap[hashlib.md5(image)] = i
print('Validation set size ' + str(len(validHashMap)))

testHashMap = {}
for i,image in enumerate(data['test_dataset']):
  testHashMap[hashlib.md5(image)] = i
print('Test set size ' + str(len(testHashMap)))


validOverlap = [];
for v in enumerate(validHashMap):
  if v in trainHashMap:
    validOverlap.append(v)

print('Validation and train sets overlap in ' + str(len(validOverlap))
  + ' elements')


testOverlap = [];
for v in enumerate(testHashMap):
  if v in trainHashMap:
    testOverlap.append(v)

print('Test and train sets overlap in ' + str(len(testOverlap))
  + ' elements')
