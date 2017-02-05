# Multinomial logistic regression trained using Stochastic Gradient Descent.
#
# For Python 2.7, install Pillow not PIL
#
# Everything is the same as multi_logistic.py but ... see comments

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

print("New training set shize ", train_dataset.shape, train_labels.shape)

# Define network
graph = tf.Graph()

### HERE THERE IS A DIFFERENCE: instead of using a big subset of 10.000
### we can use several small subsets of 128 
batch_size = 128
with graph.as_default():
  ### HERE THERE IS THE DIFFERENCE WITH non-Stochastic
  tf_train_dataset = tf.placeholder(tf.float32,
      shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)

  # Variables (the same as non-Stochastic)
  weights = tf.Variable(
      tf.truncated_normal([image_size * image_size, num_labels]))
  biases = tf.Variable(tf.zeros([num_labels]))

  # Training computation (the same as non-Stochastic)
  logits = tf.matmul(tf_train_dataset, weights) + biases
  loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
                    
  # Optimizer (the same as non-Stochastic)
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

  # Predictions for the training, validation, and test data (the as non-Stoch).
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(
      tf.matmul(tf_valid_dataset, weights) + biases)
  test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)


### HERE THERE IS A DIFFERENCE: instead of iterating not too much (since
### the sample size is big enough), we can repeat the optimization step
### 5 times more.
start = time.time()
num_steps = 3001
with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print('Initialized')
  for step in range(num_steps):
    ### HERE THERE IS THE DIFFERENCE: takes random offset
    ### It's a kind of taking random subsets (not too random, indeed!)
    ### 
    ### Parece que hay un problema de ruido en los ejemplos aleatorios.
    ###
    offset = np.random.randint(0, train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]

    _, l, predictions = session.run([optimizer, loss, train_prediction],
                                    feed_dict = {tf_train_dataset: batch_data, 
                                                 tf_train_labels: batch_labels})

    if (step % 100 == 0):
      print('Loss at step %d: %f' % (step, l))
      print('Training accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(), valid_labels))
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

end = time.time()
print('Total spent time training: ' + str(end-start))

# A sample of final execution gives us:
# Test accuracy: 86.2%
# Total spent time training: 4.29384708405

