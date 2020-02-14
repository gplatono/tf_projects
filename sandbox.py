import tensorflow as tf
import numpy as np
from tensorflow.python.client import device_lib
#from tensorflow.contrib.slim

print(device_lib.list_local_devices())

session = tf.InteractiveSession()
hello = tf.constant("Hello, world!")
#print (session.run(hello))

c1 = tf.constant([[1, 2], [3,4]], dtype=tf.float32, shape=[2, 2], name = 'x')
c2 = tf.constant([1.0, 2.0], dtype=tf.float32, shape=[1, 2], name='y')
c3 = tf.constant([3.0, 4.0, 5.0], dtype=tf.float32, shape=[3, 1], name='y1')
c4 = tf.constant([5.0, 7.0], dtype=tf.float32, shape=[2], name='y2')

print (session.run([c1, c2, c3, c4]))

op1 = tf.linalg.trace(c1)
op2 = c1 * c2

#print (op1, op2)
#print (session.run([op1, op2]))
print ("Multiply test: ", session.run(c2 * c4))

pl0_1 = tf.placeholder(tf.float32)
pl0_2 = tf.placeholder(tf.float32)
pl1 = tf.placeholder(dtype=tf.float32, shape=[2])
pl2 = tf.placeholder(dtype=tf.float32, shape=[2])
op3 = pl0_1 * pl0_2
op4 = pl1 * pl2


print (session.run(op4, {pl1: [1.0, 2], pl2: [3, 4]}))

session.close()
tf.reset_default_graph()
session = tf.InteractiveSession()


x = tf.placeholder(tf.float32, shape=[3], name='x')
w_py = np.eye(3)
w = tf.Variable(tf.convert_to_tensor(w_py, dtype=tf.float32), dtype=tf.float32, shape=[3, 3], name='w')
b = tf.Variable([1.0, 1.0, 1.0], dtype=tf.float32, shape=[3], name='b')

tf.global_variables_initializer().run()

print (session.run([tf.tensordot(w, x, axes=[[1], [0]])], feed_dict={x: [1, 2, 3]}))

y = tf.tensordot(w, x, axes=[[1], [0]]) + b
writer = tf.compat.v1.summary.FileWriter('tflogs', session.graph)
print (session.run([y], feed_dict={x: [1, 2, 3]}))
print (x, w, b, y, op1, op2)
session.close()

#tf.reset_default_graph()

"""with tf.device('/device:CPU:0'):
    # Define model parameters
    w = tf.get_variable(name='w', initializer=[.3], dtype=tf.float32)
    b = tf.get_variable(name='b', initializer=[-.3], dtype=tf.float32)
    # Define model input and output
    x = tf.placeholder(name='x', dtype=tf.float32)
    y = w * x + b

config = tf.ConfigProto()
config.log_device_placement = True

with tf.Session(config=config) as tfs:
	# initialize and print the variable y
	tfs.run(tf.global_variables_initializer())
	writer = tf.summary.FileWriter('tflogs', tfs.graph)
	print('output', tfs.run(y, {x: [1, 2, 3, 4]}))
"""
