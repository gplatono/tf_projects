from sklearn import datasets as skds
from sklearn import model_selection
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

mnist = tfds.load('mnist')

train = mnist['train']
test = mnist['test']

train = train.batch(128)
print (train.take(128))
print (len(train))
 
num_outputs = 10 # 0-9 digits
num_inputs = 784 # total pixels, 28x28
learning_rate = 0.001
epochs = 3000
batch_size = 100
#The amount of batches used for training
num_batches = int(mnist.train.num_examples / batch_size)

with tf.Session() as sess:	
	x = tf.placeholder(shape=[None, num_inputs], dtype=tf.float32, name='x')
	y = tf.placeholder(shape=[None, num_outputs], dtype=tf.float32, name='y')

	w = tf.Variable(tf.random.normal(shape=[num_inputs, num_outputs]), dtype=tf.float32, name='w')
	b = tf.Variable(tf.random.normal(shape=[num_outputs]), dtype=tf.float32, name='b')

	y_hat = tf.nn.softmax(tf.matmul(x, w) + b)
	
	loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_hat), axis=1))
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
	match = tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_hat, 1)), dtype=tf.float32)
	acc = 100 * tf.reduce_sum(match) / tf.cast(tf.size(match), tf.float32)

	tf.global_variables_initializer().run()
	for epoch in range(epochs):		
		sess.run(optimizer, feed_dict={x: X_train, y: y_train})				
		loss_epoch, acc_epoch = sess.run([loss, acc], feed_dict={x: X_train, y: y_train})
		print ("Loss: {:4.8f}, Acc: {:4.8f}".format(loss_epoch, acc_epoch))

	test_loss, test_acc = sess.run([loss, acc], feed_dict={x: X_test, y: y_test})
	print ("Test loss: {:4.8f}, Test acc: {:4.8f}".format(test_loss, test_acc))