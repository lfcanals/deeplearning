# 1-hidden layer Neural Network trained using Stochastic Gradient Descent.
#
# For Python 2.7, install Pillow not PIL
#
# Everything is the same as 1layer_nn, check the comments to see differences

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



def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])



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


train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)


# Define network
graph = tf.Graph()

batch_size = 128
beta = 0.01
with graph.as_default():
  tf_train_dataset = tf.placeholder(tf.float32,
      shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)

  weights1 = tf.Variable(
      tf.truncated_normal([image_size * image_size, 1024]))
  biases1 = tf.Variable(tf.zeros([1024]))
  weights2 = tf.Variable(
      tf.truncated_normal([1024, num_labels]))
  biases2 = tf.Variable(tf.zeros([num_labels]))
  ### THE DIFFERENCE: define a variable for dropout values
  dropout_value = tf.placeholder(tf.float32); # The value to cut values 


  # Create network
  layer1 = tf.matmul(tf_train_dataset, weights1) + biases1
  layerHidden = tf.nn.relu(layer1)
  ### THE DIFFERENCE: dropout layer
  dropoutLayer = tf.nn.dropout(layerHidden, dropout_value)
  layer2 = tf.matmul(dropoutLayer, weights2) + biases2

  loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(layer2, tf_train_labels))

  ### THE DIFFERENCE: loss adding norm2 of parameters
  lossPrime = loss + beta * (tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights2))
                    
  # Optimizer (the same as non-Stochastic but slowly movement to optimum value)
  optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(lossPrime)

  # Predictions for the training, validation, and test data (the as non-Stoch).
  train_prediction = tf.nn.softmax(layer2)
  valid_prediction = tf.nn.softmax(
    tf.matmul(
      tf.nn.relu(
          tf.matmul(tf_valid_dataset, weights1) + biases1), weights2) + biases2)
  test_prediction = tf.nn.softmax(
    tf.matmul(
      tf.nn.relu(
          tf.matmul(tf_test_dataset, weights1) + biases1), weights2) + biases2)


start = time.time()
num_steps = 3001
with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print('Initialized')
  for step in range(num_steps):
    offset = np.random.randint(0, train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]

    _, l, predictions = session.run([optimizer, loss, train_prediction],
                                    feed_dict = {tf_train_dataset: batch_data, 
                                                 tf_train_labels: batch_labels,
                                                 dropout_value: 0.5})

    if (step % 500 != 0):
      print('Loss at step %d: %f' % (step, l))
      print('Training accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(), valid_labels))
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

end = time.time()
print('Total spent time training: ' + str(end-start))

# A sample of final execution gives us:
#   For a dropout_value = 1
# Test accuracy: 92.0%
# Total spent time training: 82.0102
#
#   For a dropout_value = 0.5
# Test accuracy: 91.4%
# Total spent time training: 74.205
#
#   For a dropout_value = 0.5 and batch_size = num_trains/4, num_steps=10
# Test accuracy: 81%
# Total spent time training:  101.57
#
#   For a dropout_value = 1 and batch_size = num_trains/4, num_steps=10
# Test accuracy: 76.7%
# Total spent time training:  109.008
#
#~
#

