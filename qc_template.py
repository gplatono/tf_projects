import tensorflow.compat.v1 as tf
#from sklearn import datasets as skds
from sklearn import model_selection
import random
import numpy as np
#import matplotlib.pyplot as plt

#Generate random data
#I generate as lists and then cast into np array, but you can modify to generate random 
#np arrays directly. Something like np.random.uniform(a, b, num_samples) (look up the exact syntax)
xd = np.array([[random.uniform(-10, 10) for i in range(2)] for j in range(10000)])
yd = np.array([[float(x[0] * x[1] >= 0)] for x in xd])

#Split into train/test data
X_train, X_test, y_train, y_test = model_selection.train_test_split(xd, yd, test_size = 0.2)

#Dimensions of the layers
input_dim = xd.shape[1]
output_dim = yd.shape[1]
#Experiment with this
hidden_dim1 = 5 

#Experiment with this
learning_rate = 0.01 
epochs = 10000

with tf.Session() as sess:
	#Setting input and true output nodes
	x = tf.placeholder(shape=[None, input_dim], dtype = tf.float32)
	y = tf.placeholder(shape=[None, output_dim], dtype = tf.float32)

	#First hidden layer	params
	w1 = tf.Variable(...)
	b1 = tf.Variable(...)		
	#Experiment with this
	h1 = tf.nn.relu(...)	
	
	#You can add more layers. I got good results (99% accuracy) with two hidden layers,
	#but it should be possible to do the same with just one

	#Predicted outputs' layer
	w2 = tf.Variable(...)
	b2 = tf.Variable(...)
	y_hat = tf.nn.sigmoid(...)

	#Infrastructure code
	loss = tf.reduce_mean(tf.square(y - y_hat))

	match = tf.cast(tf.math.equal(tf.round(y_hat), y), dtype = tf.float32)
	acc = tf.math.reduce_sum(match) / tf.to_float(tf.size(match))

	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
	tf.global_variables_initializer().run()

	for epoch in range(epochs):
		#Run the training by feeding the training data to optimizer
		sess.run(optimizer, feed_dict = {x: X_train, y: y_train})
		#Print something here to track the progress of learning
	

	print("Test accuracy: ", sess.run(acc, feed_dict = {x: X_test, y: y_test}))
	
	#I suggest to print here a few values of the predictions "yhat" and true values "y_test",
	#to compare them, and see what you actually get