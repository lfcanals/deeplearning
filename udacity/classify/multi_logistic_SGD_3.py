# Multinomial logistic regression trained using Stochastic Gradient Descent.
#
# For Python 2.7, install Pillow not PIL
#
# This code is the same as multi_logistic_SGD_2 but the working set of images
# is different: taken from a Python library.
#
# Accuracy is better than with the self-made set of images.

from __future__ import print_function
import numpy as np
from scipy import ndimage
from six.moves import cPickle as pickle
from six.moves import range
import tensorflow as tf
import time

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets 
tf.set_random_seed(0)

image_size = 28
num_labels = 10
np.random.seed(133)



mnist = read_data_sets("data", one_hot=True, reshape=False, validation_size=0)



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
    train_X, train_Y = mnist.train.next_batch(batch_size)
    train_X = train_X[:,:,:,0]
    
    train_dict = { X: train_X, Y_:train_Y }

    # Aqui antes habia otro parametro:
    #session.run([optimizer, loss, train_prediction], feed_dict = feed_dict);
    session.run([optimizer, loss], feed_dict = train_dict)


  # Success in test data
  test_X = mnist.test.images[:,:,:,0]
  test_Y = mnist.test.labels

  exec_dict = { X: test_X, Y_: test_Y }
  resultAcc, resultCrossEntr = session.run([acc, loss], feed_dict = exec_dict)
  print('Test accuracy: %.1f%%' % (resultAcc*100))


# Final Test accuracy: 92.3% (3001 steps of 0.003)

