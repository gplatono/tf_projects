import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

#Set the parameters of the model and learning protocol
num_outputs = 10 # digits 0-9
num_inputs = 28 # 28x28 pixels
learning_rate = 0.001
epochs = 30
batch_size = 256

#Get the dataset and the accompaniying description
mnist, mnist_info = tfds.load('mnist', with_info=True)
print (mnist_info)

train, test = mnist['train'], mnist['test']
print (train.element_spec)

#Reshape and cast to the types supported by the sparse categorical cross-entropy loss
train = train.map(lambda x: {'image': tf.reshape(x['image'], shape=[x['image'].shape[0], x['image'].shape[1]]), 'label': x['label']})
train = train.map(lambda x: {'image': tf.cast(x['image'], tf.float32), 'label': tf.cast(x['label'], tf.int32)})
train.cache()

test = test.map(lambda x: {'image': tf.reshape(x['image'], shape=[x['image'].shape[0], x['image'].shape[1]]), 'label': x['label']})
test = test.map(lambda x: {'image': tf.cast(x['image'], tf.float32), 'label': tf.cast(x['label'], tf.int32)})
test.cache()

#Divide the datasets into batches. We only use first 1000 samples for testing
train = train.batch(batch_size)
test = test.batch(1000).take(1)

class Model():
	def __init__(self, input_shape, output_shape, layer_shapes=[]):
		self.shapes = [input_shape] + layer_shapes + [output_shape]
		self.w = []
		self.b = []
		for i in range(1, len(self.shapes)):
			self.w.append(tf.Variable(tf.random.normal(shape=self.shapes[i-1] + self.shapes[i]), dtype=tf.float32, name='w{}'.format(i)))
			self.b.append(tf.Variable(tf.random.normal(shape=self.shapes[i]), dtype=tf.float32, name='b{}'.format(i)))
		self.trainable_parameters = self.w + self.b
		
	@tf.function
	def predict(self, x):
		"""Run forward inference on the model. """
		out = x
		for i in range(len(self.w)):
			contr_dims = len(self.shapes[i])			
			f_contr =[j for j in range(1, contr_dims+1)]
			s_contr =[j for j in range(0, contr_dims)]
			out = tf.tensordot(out, self.w[i], axes=[f_contr, s_contr]) + self.b[i]
			if i < len(self.w) - 1:
				out = tf.sigmoid(out)				
		return out

	def get_trainable_parameters(self):
		"""Return the trainable parameters that will be used in optimization. """
		return self.trainable_parameters

	@tf.function
	def eval(self, x, y):
		"""Evaluate the model. Run forward inference and return the prediction, loss and accuracy. """
		y_hat = self.predict(x)

		#Note that the loss here takes inputs as y = 3 (int32), y_hat = [0.0, 0.0, 1.0, ..., 0.0] (one-hot of float32)
		loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_hat))
		pred_match = tf.cast(tf.equal(tf.cast(tf.argmax(y_hat, 1), dtype=tf.int32), y), dtype=tf.float32)
		accuracy = 100 * tf.reduce_sum(pred_match) / tf.cast(tf.size(pred_match), dtype=tf.float32)
		return y_hat, loss, accuracy

	def info(self):
		"""Return the description of the model. """
		from functools import reduce
		ret_val = "MODEL INFO:\n"				
		ret_val += "Input Shape: {}\n".format('[' + ','.join([str(item) for item in self.shapes[0]]) + ']')
		for i in range(1, len(self.shapes)):				
			params = reduce(lambda x, y: x * y, self.w[i-1].shape) + self.b[i-1].shape[0]
			ret_val += "Layer {}. Shape: {}; Total Parameters: {};\n".format(i, '[' + ','.join([str(item) for item in self.shapes[i]]) + ']', params)
		return ret_val
	
	def train(self, train_data, epochs, optimizer, learning_rate=0.001):
		"""Train the model using the provided dataset and optimizer object. """
		print ("Training in progress...")
		for epoch in range(epochs):
			print ("Epoch: {}".format(epoch+1))
			avg_loss = 0.0
			avg_acc = 0.0
			batch_num = 0
			for batch in train_data:		
				with tf.GradientTape(persistent=True) as tape:
					x, y = batch['image'], batch['label']
					tape.watch(x)
					_, loss, acc = self.eval(x, y)			
					avg_loss += loss.numpy()
					avg_acc += acc.numpy()
					batch_num += 1
				
				#Compute the gradient of the loss w.r.t trainable parameters and update them
				gradient = tape.gradient(loss, self.get_trainable_parameters())
				optimizer.apply_gradients(zip(gradient, self.get_trainable_parameters()))

			avg_loss /= batch_num
			avg_acc /= batch_num
			print ("Average batch loss: {:.2f}, Average batch accuracy: {:.2f}%".format(avg_loss, avg_acc))
		print ("Training complete. Run the eval() method to test the performance on the testing dataset.")

#Create the model and optimizer, then run the training, and test on the first 1000 samples
mlp = Model(input_shape=[num_inputs, num_inputs], output_shape = [num_outputs], layer_shapes = [[10]])
print(mlp.info())
optimizer = tf.optimizers.Adam(learning_rate = learning_rate)

mlp.train(train_data = train, epochs = epochs, optimizer = optimizer, learning_rate = learning_rate)

for batch in test:
	x, y = batch['image'], batch['label']
	_, test_loss, test_acc = mlp.eval(x, y)
	print ("\nTest loss: {:.2f}, Test accuracy: {:.2f}%\n".format(test_loss, test_acc))