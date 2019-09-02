import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.client import device_lib
import numpy

#graph = tf.Graph()
#tf.compat.v1.reset_default_graph()

def add_layer(x, id, in_dim, out_dim):
	#print (x.shape, in_dim, out_dim)
	#print (tf.random_normal(shape=[out_dim, in_dim]))
	w = tf.get_variable('w_' + str(id), shape=(out_dim, in_dim), initializer=tf.random_normal_initializer())
	b = tf.get_variable('b_' + str(id), shape=(out_dim), initializer=tf.random_normal_initializer())
	net = tf.tensordot(w, x, axes=[1, 0]) + b	
	return tf.sigmoid(net)

mnist = input_data.read_data_sets(os.path.join('.', 'mnist'), one_hot=True)
X_train = tf.convert_to_tensor(mnist.train.images)
Y_train = tf.convert_to_tensor(mnist.train.labels)
X_test = tf.convert_to_tensor(mnist.test.images)
Y_test = tf.convert_to_tensor(mnist.test.labels)

nbatch = 10
batchsize = -(-X.train.shape[0] // nbatch)
lrate = 0.1
in_dim = X_train.shape
hdim1 = 50
out_dim = 10

x = tf.placeholder(shape=[in_dim], dtype=tf.float32)
l1 = add_layer(x, 1, in_dim, hdim1)
output = add_layer(l1, 2, hdim1, out_dim)

# #output = tf.compat.v1.Variable(tf.random.normal(shape=[10], mean=0, stddev=1.0), dtype=tf.float32)
loss_op = tf.losses.mean_squared_error(labels = Y_train, predictions = output)
train_op = tf.train.AdamOptimizer(learning_rate = lrate).minimize(loss_op)

# session = tf.Session()
# tf.global_variables_initializer().run(session=session)
# result = session.run([tf.ones(shape=[10, 10], dtype=tf.float32, name='1s')])
# print (result)
# session.close()

#print (X_test)

#print (device_lib.list_local_devices())