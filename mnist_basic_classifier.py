import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.keras.datasets import mnist

#Load and preprocess the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
y_train = np.eye(10)[y_train.reshape(-1)]
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
y_test = np.eye(10)[y_test.reshape(-1)]

num_outputs = 10 # digits 0-9
num_inputs = 784 # 28x28 pixels
learning_rate = 0.001
epochs = 100
batch_size = 300
data_len = x_train.shape[0] #Number of inputs in the dataset

with tf.Session() as sess:
	x = tf.placeholder(shape=[None, num_inputs], dtype=tf.float32, name='x')
	y = tf.placeholder(shape=[None, num_outputs], dtype=tf.float32, name='y')

	w = tf.Variable(tf.random.normal(shape=[num_inputs, num_outputs], stddev = 0.01), dtype=tf.float32, name='w')
	b = tf.Variable(tf.random.normal(shape=[num_outputs], stddev = 0.01), dtype=tf.float32, name='b')

	y_hat = tf.nn.softmax(tf.sigmoid(tf.matmul(x, w) + b))
		
	loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_hat), axis=1))
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
	match = tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_hat, 1)), dtype=tf.float32)
	acc = 100 * tf.reduce_sum(match) / tf.cast(tf.size(match), tf.float32)

	tf.global_variables_initializer().run()

	for epoch in range(epochs):
		batch_idx = 0

		#Total avg loss/acc for training data
		avg_loss = avg_acc = 0

		#Iterate over batches and run the optimization & compute loss/acc per batch
		while batch_idx * batch_size < data_len:
			start = batch_size * batch_idx
			end = batch_size * (batch_idx + 1)

			sess.run(optimizer, feed_dict={x: x_train[start: end], y: y_train[start: end]})			

			batch_loss, batch_acc = sess.run([loss, acc], feed_dict={x: x_train[start: end], y: y_train[start: end]})			
			avg_loss += batch_loss
			avg_acc += batch_acc
			batch_idx += 1

		avg_loss /= batch_idx
		avg_acc /= batch_idx
		print ("Epoch: {}, Loss: {:.2f}, Acc: {:.2f}%".format(epoch + 1, avg_loss, avg_acc))
			
	test_loss, test_acc = sess.run([loss, acc], feed_dict={x: x_test, y: y_test})
	print ("Test loss: {:.2f}, Test acc: {:.2f}%".format(test_loss, test_acc))