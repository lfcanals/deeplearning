# Multinomial logistic regression trained using Stochastic Gradient Descent.
#
# For Python 2.7, install Pillow not PIL
#
#
#
# Everything is the same as multi_logistic_SGD.py but adapted to be
# compatible easly with the example from Google Youtube course and
# image set as Python library tensorflow.contrib.learn.python.learn.datasets
# .mnist.read_data_sets


from __future__ import print_function
import numpy as np
from scipy import ndimage
from six.moves import cPickle as pickle
from six.moves import range
import tensorflow as tf
import time


pickle_file = 'notMNIST.pickle'
image_size = 28
num_labels = 10
np.random.seed(133)




f = open(pickle_file, 'r')
data = pickle.load(f)
train_dataset = data['train_dataset']
train_labels = data['train_labels']
valid_dataset = data['valid_dataset']
valid_labels = data['valid_labels']
test_dataset = data['test_dataset']
test_labels = data['test_labels']
del data
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

train_labels = (np.arange(num_labels) == train_labels[:,None]).astype(np.float32)
test_labels = (np.arange(num_labels) == test_labels[:,None]).astype(np.float32)

# Define network
graph = tf.Graph()

### HERE THERE IS A DIFFERENCE: instead of using a big subset of 10.000
### we can use several small subsets of 128 
batch_size = 128
with graph.as_default():
  X = tf.placeholder(tf.float32, [None, image_size, image_size])
  W = tf.Variable(tf.zeros([image_size * image_size, num_labels]))
  b = tf.Variable(tf.zeros([num_labels]))

  logits = tf.matmul(tf.reshape(X, [-1, image_size*image_size]), W) + b

  # Equivalent to loss = tf.reduce_mean(
  #     tf.nn.softmax_cross_entropy_with_logits(logits, Y_))      
  Y = tf.nn.softmax(logits)

  Y_ = tf.placeholder(tf.float32, [None, num_labels]) 
  loss = -tf.reduce_mean(Y_ * tf.log(Y)) * 1000.0

  optimizer = tf.train.GradientDescentOptimizer(0.003).minimize(loss)

  ## Testing
  is_correct = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
  acc = tf.reduce_mean(tf.cast(is_correct, tf.float32))

  
start = time.time()
num_steps = 3001
with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print('Initialized')
  for step in range(num_steps):
    offset = np.random.randint(0, train_labels.shape[0] - batch_size)
    train_X = train_dataset[offset:(offset + batch_size), :, :]
    train_Y = train_labels[offset:(offset + batch_size)]

    train_dict = { X: train_X, Y_:train_Y }

    # Aqui antes habia otro parametro:
    #session.run([optimizer, loss, train_prediction], feed_dict = feed_dict);
    session.run([optimizer, loss], feed_dict = train_dict)


  # Success in test data
  test_X = test_dataset
  test_Y = test_labels
  exec_dict = { X: test_X, Y_: test_Y }
  resultAcc, resultCrossEntr = session.run([acc, loss], feed_dict = exec_dict)
  print('Test accuracy: %.1f%%' % (resultAcc*100))
