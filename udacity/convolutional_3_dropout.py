# Multinomial logistic regression trained using Stochastic Gradient Descent.
#
# For Python 2.7, install Pillow not PIL
#
# This code is the same as convolutional_3 but with wider convolution
# and a droput.
#
#
#
# . . .  . . . . .      (input data, 1-deep)             X [batch, 28, 28, 1]
# @ @ @  @ @ @ @ @   -- conv. layer 5x5x1=>4 stride 1    W1 [6,6,1,6] + B1[5]
# ::::::::::::::::                                       Y1 [batch,28,28,4]
#   @ @  @ @ @ @     -- conv. layer 5x5x4=>8 stride 2    W2 [5,5,6,12] +  B2[12]
#   ::::::::::::                                         Y2 [batch,14,14,8]
#     @  @ @ @       -- conv. layer 4x4x8=>12 stride 2   W3 [4,4,12,24] + B3[24]
#     :::::::                                            Y3 [batch, 7, 7, 24] 
#                                         => reshaped to YY [batch, 7*7*24]
#      \\x\x/        -- fully connected layer (relu)     W4 [7*7*24,200]+B4[200]
#        . .                                             Y4 [batch, 200]
#       x\x/         -- fully connected layer (softmax)  W5 [200, 10]           B5 [10]
#       . .                                              Y [batch, 20]


import numpy as np
import tensorflow as tf
import math
import time

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets 
tf.set_random_seed(0)

image_size = 28
num_labels = 10
batch_size = 128


np.random.seed(133)
mnist = read_data_sets("data", one_hot=True, reshape=False, validation_size=0)


# variable learning rate
lr = tf.placeholder(tf.float32)
# Probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.75 at training time
pkeep = tf.placeholder(tf.float32)

# Network
X = tf.placeholder(tf.float32, [None, image_size, image_size, 1])

W1 = tf.Variable(tf.truncated_normal([6,6,1,6], stddev=0.1))
b1 = tf.Variable(tf.ones([6])/10)

W2 = tf.Variable(tf.truncated_normal([5,5,6,12], stddev=0.1))
b2 = tf.Variable(tf.ones([12])/10)

W3 = tf.Variable(tf.truncated_normal([4,4,12,24], stddev=0.1))
b3 = tf.Variable(tf.ones([24])/10)


W4 = tf.Variable(tf.truncated_normal([7*7*24, 200], stddev=0.1))
b4 = tf.Variable(tf.ones([200])/10)

W5 = tf.Variable(tf.truncated_normal([200, num_labels], stddev=0.1))
b5 = tf.Variable(tf.ones([num_labels])/10)


Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='SAME') + b1)
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1,2,2,1], padding='SAME') + b2)
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1,2,2,1], padding='SAME') + b3)

YY = tf.reshape(Y3, shape=[-1, 7*7*24])

Y4 = tf.nn.relu(tf.matmul(YY, W4) + b4)
YY4 = tf.nn.dropout(Y4, pkeep)
Y5 = tf.matmul(YY4, W5) + b5
logits = Y5
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
    # delme please... train_X = train_X[:,:,:,:]

    # learning rate decay
    max_learning_rate = 0.003
    min_learning_rate = 0.0001
    decay_speed = 2000.0
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)

    # the backpropagation training step
    sess.run(optimizer, {X: train_X, Y_: train_Y, pkeep: 0.75, 
                         lr: learning_rate, pkeep: 0.75})

    if(i%100==0): print('*');

print('\n');

# Success in test data
# delmeplease test_X = mnist.test.images[:,:,:,0]
test_X = mnist.test.images
test_Y = mnist.test.labels

exec_dict = { X: test_X, Y_: test_Y }
a, c = sess.run([acc, cross_entropy], {X: test_X, Y_: test_Y, pkeep: 1.0})
print('Test accuracy: %.1f%%' % (a*100))


# Final accuracy : Test accuracy: 99.1% 3001 steps using decay rate step

