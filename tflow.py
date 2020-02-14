import tensorflow as tf
import keras as k


# x = tf.constant(2.0)
# y = tf.constant(3.0)

# z = x * y

# c = tf.Variable([1.0, 2.0], shape=[2], name="myvar", dtype=tf.float32)

# x = tf.compat.v1.placeholder(tf.float32, shape=[2], name="input")
# y = tf.compat.v1.placeholder(tf.float32, shape=[1], name="output")

sess1 = tf.compat.v1.InteractiveSession()

g = tf.Graph()

model = k.Sequential()
model.add(k.layers.Dense(10, input_shape=(256, ), activation='tanh'))
model.add(k.layers.Dense(100, activation='relu'))

with g.as_default():
	x = tf.placeholder(tf.float32, shape = (1,4))
	w = tf.Variable([1, 1, 1, 1], shape = (4,), dtype=tf.float32)
	b = tf.Variable(3.0, dtype=tf.float32)

	c = tf.constant([1, 2, 3, 4], shape = (1, 4), dtype=tf.float32)
	d = tf.constant([1, 2, 3, 4], shape = (1, 4), dtype=tf.float32)
	d1 = tf.reshape(d, shape=(2, 2))

	y = tf.matmul(x, tf.reshape(w, shape=(4, 1))) + b

with tf.Session(graph = g) as sess:	
	#print (sess.run(tf.matmul(d, tf.matrix_transpose(c))))
	
	tf.global_variables_initializer().run()

	print (sess.run(y, {x:[[1, 0, 0, 0]]}))	
	print (sess.run(d1))	

# print (tf.dtypes.as_dtype('int32'))
# print (tf.int8.min)

sess1.close()