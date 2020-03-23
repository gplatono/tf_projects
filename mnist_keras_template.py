import numpy as np
import tensorflow as tf

#Set the parameters of the model and learning protocol
num_outputs = 10 # digits 0-9
num_inputs = 28 * 28 # 28x28 pixels
learning_rate = 0.001
epochs = 30
batch_size = 256

#Load and preprocess the data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = tf.constant(np.reshape(x_train, newshape=(x_train.shape[0], 784)), dtype=tf.float32)
x_test = tf.constant(np.reshape(x_test, newshape=(x_test.shape[0], 784)), dtype=tf.float32)
y_train = tf.one_hot(y_train, num_outputs)
y_test = tf.one_hot(y_test, num_outputs)

print ("Input shapes: ", x_train.shape, y_train.shape)

class Model(tf.keras.Model):
	def __init__(self, hidden_dims=[10]):
		super(Model, self).__init__()
		pass
		
	def call(self, x):
		"""Run forward inference on the model. """
		pass

mlp = Model(hidden_dims=[10, 10])

#Fill in the model compilation and training here

loss, acc = mlp.evaluate(x_test, y_test, verbose = 0)
print ("\nTest loss: {:.2f}, Test accuracy: {:.2f}%\n".format(loss, 100 * acc))