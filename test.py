import tensorflow as tf
from sklearn import datasets as skds
import random

x_train = [[random.uniform(-100, 100) for i in range(2)] for j in range(200)]
y_train = []
for x in x_train:
	y_train.append(float(x[0] * x[1] >= 0))	

#print (x_train, y_train)
input_dim = x_train.shape[1]
output_dim = y_train.shape[1]

print (sum(y_train))
with tf.compat.v1.Session() as sess:
	X = tf.compat.v1.placeholder(shape=[None, input_dim], dtype=tf.float32)
	Y = tf.compat.v1.placeholder(shape=[None, output_dim], dtype=tf.float32)
	#W = tf.compat.v1.Variable(shape=[])