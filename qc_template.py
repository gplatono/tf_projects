#from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow.compat.v1 as tf
from sklearn import datasets as skds
from sklearn import model_selection
import random
import numpy as np
import  matplotlib.pyplot as plt

xd = [[random.uniform(-10, 10) for i in range(2)] for j in range(10000)]
yd = []
for item in xd:
	yd.append([float(item[0] * item[1] >= 0)])
xd = np.array(xd)
yd = np.array(yd)

# for i in range(20):
# 	print (xd[i], yd[i])
# plt.figure(figsize =(14 ,8))
# plt.plot(xd,yd,'b.')
# plt.title('Original  Dataset ')
# plt.show()

X_train, X_test, y_train, y_test = model_selection.train_test_split(xd, yd, test_size = 0.2)

input_dim = xd.shape[1]
output_dim = yd.shape[1]
#Experiment with this
hidden_dim1 = 5 
#hidden_dim2 = 5

#Experiment with this
learning_rate = 0.01 

#Experiment with this
epochs = 10000

with tf.Session() as sess:
	x = tf.placeholder(shape=[None, input_dim], dtype = tf.float32)
	y = tf.placeholder(shape=[None, output_dim], dtype = tf.float32)
	
	w1 = tf.Variable(...)
	b1 = tf.Variable(...)		

	#Experiment with this
	h1 = tf.nn.relu(...)
	
	#Experiment with this
	# w2 = tf.Variable(...)
	# b2 = tf.Variable(...)
	# h2 = tf.nn.relu(...)

	w3 = tf.Variable(...)
	b3 = tf.Variable(...)
	y_hat = tf.nn.sigmoid(...)


	#infrastructure code
	loss = tf.reduce_mean(tf.square(y - y_hat))
	match = tf.cast(tf.math.equal(tf.round(y_hat), y), dtype = tf.float32)
	myacc = tf.math.reduce_sum(match) / tf.to_float(tf.size(match))
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
	tf.global_variables_initializer().run()


	for epoch in range(epochs):
		sess.run(optimizer, feed_dict = {x: X_train, y: y_train})

		#print something to track the progress of learning
	

	print("Test acc: ", sess.run(myacc, feed_dict = {x: X_test, y: y_test}))
	
	#I suggest to print here a few values of the predictions "yhat" and true values "y_test",
	#to compare them, and see what you actually get