import tensorflow.compat.v1 as tf
#from sklearn import datasets as skds
from sklearn import model_selection
import random
import numpy as np
#import  matplotlib.pyplot as plt

#xd, yd = skds.make_regression(n_samples = 20000,  n_features = 1,  n_informative = 2,  n_targets=1, noise= 1.0)

#if (yd.ndim == 1):
#	yd = yd.reshape(len(yd) ,1)
xd = np.array([[random.uniform(-10, 10) for i in range(2)] for j in range(10000)])
yd = np.array([[float(x[0] * x[1] >= 0)] for x in xd])

# for i in range(20):
# 	print (xd[i], yd[i])
# plt.figure(figsize =(14 ,8))
# plt.plot(xd,yd,'b.')
# plt.title('Original  Dataset ')
# plt.show()

X_train, X_test, y_train, y_test = model_selection.train_test_split(xd, yd, test_size = 0.2)

input_dim = xd.shape[1]
output_dim = yd.shape[1]
hidden_dim1 = 5
hidden_dim2 = 5
learning_rate = 0.01
epochs = 10000

with tf.Session() as sess:
	x = tf.placeholder(shape=[None, input_dim], dtype = tf.float32)
	y = tf.placeholder(shape=[None, output_dim], dtype = tf.float32)

	# w1 = tf.Variable(tf.random.normal(shape=[input_dim, output_dim]))
	# b1 = tf.Variable(tf.random.normal(shape=[output_dim]))	
	# y_hat = tf.matmul(x, w1) + b1
	
	w1 = tf.Variable(tf.random.normal(shape=[input_dim, hidden_dim1], stddev = 1.0))
	b1 = tf.Variable(tf.random.normal(shape=[hidden_dim1], stddev = 1.0))		
	h1 = tf.nn.relu(tf.matmul(x, w1) + b1)
	
	w2 = tf.Variable(tf.random.normal(shape=[hidden_dim1, hidden_dim2], stddev = 1.0))
	b2 = tf.Variable(tf.random.normal(shape=[hidden_dim2], stddev = 1.0))
	h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

	w3 = tf.Variable(tf.random.normal(shape=[hidden_dim2, output_dim], stddev = 1.0))
	b3 = tf.Variable(tf.random.normal(shape=[output_dim], stddev = 1.0))
	y_hat = tf.nn.sigmoid(tf.matmul(h2, w3) + b3)

	loss = tf.reduce_mean(tf.square(y - y_hat))
	match = tf.cast(tf.math.equal(tf.round(y_hat), y), dtype = tf.float32)
	myacc = tf.math.reduce_sum(match) / tf.to_float(tf.size(match))
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
	tf.global_variables_initializer().run()
	for epoch in range(epochs):
		sess.run(optimizer, feed_dict = {x: X_train, y: y_train})
		if (epoch % 50 == 0):
			print("current loss: ", sess.run(loss, feed_dict = {x: X_train, y: y_train}))
			print("current acc: ", sess.run(myacc, feed_dict = {x: X_train, y: y_train}))		
	#w1_hat, b1_hat, w2_hat, b2_hat = sess.run([w1, b1, w2, b2])
	print ("Test loss: ", sess.run(loss, feed_dict = {x: X_test, y: y_test}))
	print("Test acc: ", sess.run(myacc, feed_dict = {x: X_test, y: y_test}))
	yhattest = sess.run(y_hat, feed_dict = {x: X_test, y: y_test})
	for i in range(20):
	 	print (yhattest[i], y_test[i])