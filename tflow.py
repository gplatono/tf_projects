from sklearn import datasets as skds
from sklearn import model_selection
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow as TF

mnist = TF.examples.tutorials.mnist.input_data.read_data_sets(os.path.join(datasetslib.datasets_root, 'mnist'), one_hot=True)

X, y = skds.make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=1)
  
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=.4, random_state=42)

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

	y_pred = sess.run(tf.argmax(y_hat, 1), feed_dict={x: X_test})
	y_orig = sess.run(tf.argmax(y, 1), feed_dict={y: y_test})

#Plot the original and predicted labels
plt.figure(figsize=(14, 4))
plt.subplot(1, 2, 1)
plt.scatter(X_test[:, 0], X_test[:, 1], marker='o', c=y_orig)
plt.title('Original')
plt.subplot(1, 2, 2)
plt.scatter(X_test[:, 0], X_test[:, 1], marker='o', c=y_pred)
plt.title('Predicted')
plt.show()