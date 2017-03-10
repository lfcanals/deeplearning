# 1-hidden layer Neural Network trained using Stochastic Gradient Descent.
#
# For Python 2.7, install Pillow not PIL
#
# LeNet5 implementation for MNIST resolution

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
num_channels = 1 # Grayscale

np.random.seed(133)

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size, image_size, num_channels))\
        .astype(np.float32)
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


train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)


print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

# Define network
graph = tf.Graph()

batch_size = 16
patch_size = 5
depth = 5
beta = 0.01
num_hidden = 64
num_hidden2 = 32
with graph.as_default():
  tf_train_dataset = tf.placeholder(tf.float32,
      shape=(batch_size, image_size, image_size, num_channels))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)

  weights1 = tf.Variable(
      tf.truncated_normal([patch_size, patch_size, num_channels, depth], 
        stddev=0.1))
  biases1 = tf.Variable(tf.zeros([depth]))


  weights2 = tf.Variable(
      tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
  biases2 = tf.Variable(tf.constant(1.0, shape=([depth])))

  weights3 = tf.Variable(
      tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
  biases3 = tf.Variable(tf.constant(1.0, shape=([depth])))


  weights4 = tf.Variable(
      tf.truncated_normal([image_size // 4 * image_size // 4 * depth, 
        num_hidden], stddev=0.1))
  biases4 = tf.Variable(tf.constant(1.0, shape=([num_hidden])))

  weights5 = tf.Variable(tf.truncated_normal(
        [num_hidden, num_hidden2], stddev=0.1))
  biases5 = tf.Variable(tf.constant(1.0, shape=[num_hidden2]))

  weights6 = tf.Variable(tf.truncated_normal(
        [num_hidden2, num_labels], stddev=0.1))
  biases6 = tf.Variable(tf.constant(1.0, shape=[num_labels]))


  dropout_value = tf.placeholder(tf.float32); # The value to cut values 


  
  def model(data):
    '''
        Defines the model
    '''
    conv = tf.nn.conv2d(data, weights1, [1,2,2,1], padding='SAME')
    hidden = tf.nn.relu(conv + biases1)

    #pooling1 = tf.nn.max_pool(hidden, ksize=[1,2,2,1], strides=[1,2,2,1],
    #    padding='SAME', name='pooling1')
    #norm1 = tf.nn.lrn(pooling1, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
    #    name='norm1')
    norm1 = hidden

    conv = tf.nn.conv2d(norm1, weights2, [1,2,2,1], padding='SAME')
    hidden = tf.nn.relu(conv + biases2)

    #pooling2 = tf.nn.max_pool(hidden, ksize=[1,2,2,1], strides=[1,2,2,1],
    #    padding='SAME', name='pooling2')
    #norm2 = tf.nn.lrn(pooling2, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
    #    name='norm2')
    norm2 = hidden

    conv = tf.nn.conv2d(norm2, weights3, [1,1,1,1], padding='SAME')
    hidden = tf.nn.relu(conv + biases3)

    shape = hidden.get_shape().as_list()
    reshape = tf.reshape(hidden, [shape[0], shape[1]*shape[2]*shape[3]])

    hidden = tf.nn.relu(tf.matmul(reshape, weights4) + biases4)
    hidden = tf.matmul(hidden, weights5) + biases5

    hidden = tf.nn.dropout(hidden, dropout_value)

    result = tf.matmul(hidden, weights6) + biases6

    return result






  train_model = model(tf_train_dataset)
  loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(train_model, tf_train_labels)) \
      + beta * (tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights2) \
                + tf.nn.l2_loss(weights3) + tf.nn.l2_loss(weights4) \
                + tf.nn.l2_loss(weights5) + tf.nn.l2_loss(weights6))
                    
  global_step = tf.Variable(0, trainable=False)
  learning_rate = tf.train.exponential_decay(0.1, global_step, 100000, 
        0.96, staircase=True)

  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

  # Predictions for the training, validation, and test data (the as non-Stoch).
  train_prediction = tf.nn.softmax(train_model)
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
  test_prediction = tf.nn.softmax(model(tf_test_dataset))

start = time.time()
num_steps = 10001
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
                                                 dropout_value: 0.5 })

    if (step % 500 == 0):
      print('Loss at step %d: %f' % (step, l))
      print('Training accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(
                feed_dict = { dropout_value : 1.0} ), valid_labels))
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval( 
        feed_dict = { dropout_value: 1.0 }), test_labels))

end = time.time()
print('Total spent time training: ' + str(end-start))

# A sample of final execution gives us:
#  With no pooling:
#     Test accuracy: 90.4%
#     Total spent time training: 64.7587709427
#
#  With pooling:
#     Test accuracy: 86.7%
#     Total spent time training: 206.296



