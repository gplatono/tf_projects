#import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
#from tensorflow.keras.datasets import mnist


Get the dataset and the accompaniying description
mnist, mnist_info = tfds.load('mnist', with_info=True)

print (mnist_info)

# train = mnist['train']
# test = mnist['test']

# train = train.batch(256)
# test = test.batch(1000).take(1)

# #elem = test.take(1000)
# for item in test:
# 	X_test = item['image']
# 	y_test = item['label']
# 	print (y_test)


#iter = train.make_one_shot_iterator()

#X, y = iter.get_next()
# for item in train:
# 	X_train = item['image']
# 	y_train = item['label']
# 	print ("IMAGE: ", X_train)
# 	print ("LABEL: ", y_train)
@tf.function
def predict(inputs):
	w = tf.Variable(tf.random.normal(shape=[num_inputs, num_inputs, num_outputs]), dtype=tf.float32, name='w')
	b = tf.Variable(tf.random.normal(shape=[num_outputs]), dtype=tf.float32, name='b')

	y_hat = tf.nn.softmax(tf.tensordot(x, w, axes=[[1, 2], [0, 1]]) + b)
	return y_hat

@tf.function
def train(train_data):
	for item in train_data:
		x_train = item['image']
		y_train = item['label']

num_outputs = 10 # digits 0-9
num_inputs = 784 # 28x28 pixels
learning_rate = 0.001
epochs = 100
batch_size = 300
data_len = x_train.shape[0]

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
		avg_loss = avg_acc = 0
		while batch_idx * batch_size < data_len:
			start = batch_size * batch_idx
			end = batch_size * (batch_idx + 1)
			#print (sess.run(tf.matmul(x, w) + b, feed_dict={x: x_train[start: start+1], y: y_train[start: start+1]}))
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