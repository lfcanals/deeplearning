# Multinomial logistic regression trained using Stochastic Gradient Descent.
#
# For Python 2.7, install Pillow not PIL
#
# This code is the same as leNet5 but the working set of images
# is different: taken from a Python library.
#
# Accuracy is better than with the self-made set of images.
import numpy as np
import tensorflow as tf
import math
import time

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets 
tf.set_random_seed(0)

image_size = 28
num_labels = 10
patch_size = 5



batch_size = 128


np.random.seed(133)
mnist = read_data_sets("data", one_hot=True, reshape=False, validation_size=0)


# variable learning rate
lr = tf.placeholder(tf.float32)
# Probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.75 at training time
pkeep = tf.placeholder(tf.float32)

# Network
X = tf.placeholder(tf.float32, [None, image_size, image_size])

W1 = tf.Variable(tf.zeros([image_size * image_size, 1024]))
b1 = tf.Variable(tf.ones([1024])/10)
W2 = tf.Variable(tf.zeros([1024, num_labels]))
b2 = tf.Variable(tf.zeros([num_labels]))

### Todo: create the network...
W3 = tf.Variable(tf.zeros([image_size * image_size, 1024]))
W4 = tf.Variable(tf.zeros([image_size * image_size, 1024]))
W5 = tf.Variable(tf.zeros([image_size * image_size, 1024]))

Y1 = tf.matmul(tf.reshape(X, [-1, image_size*image_size]), W1) + b1
layerHidden = tf.nn.dropout(Y1, pkeep)
Y2 = tf.matmul(layerHidden, W2) + b2
logits = Y2
Y = tf.nn.softmax(logits)

# Measures
Y_ = tf.placeholder(tf.float32, [None, num_labels])
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*batch_size

is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
acc = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# Optimizer
optimizer = tf.train.AdamOptimizer(lr).minimize(cross_entropy)


init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

print('Training')
num_steps=3001
for i in range(num_steps):
    train_X, train_Y = mnist.train.next_batch(batch_size)
    train_X = train_X[:,:,:,0]

    # learning rate decay
    max_learning_rate = 0.003
    min_learning_rate = 0.0001
    decay_speed = 2000.0
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)

    # the backpropagation training step
    sess.run(optimizer, {X: train_X, Y_: train_Y, pkeep: 0.75, lr: learning_rate})



# Success in test data
test_X = mnist.test.images[:,:,:,0]
test_Y = mnist.test.labels

exec_dict = { X: test_X, Y_: test_Y }
a, c = sess.run([acc, cross_entropy], {X: test_X, Y_: test_Y, pkeep: 1.0})
print('Test accuracy: %.1f%%' % (a*100))


# Final accuracy : Test accuracy: 92.2% 3001 steps using decay rate step

