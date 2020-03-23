import numpy as np
import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing as skpp
import zipfile

words = zipfile.ZipFile('text8.zip').read('text8').split()
print (words[0:10])

wid = 0
w2id = {}
id2w = {}
for word in words:
	if word not in w2id:
		w2id[word] = wid
		id2w[wid] = word
		wid += 1

wids = [w2id[word] for word in words]

def load_data(words):
	cbow = skipgrams = []
	from os import path
	if path.exists('text8_cbow.npy'):
		cbow = np.load('text8_cbow.npy', allow_pickle=True)
	else:
		cbow = np.array([[[w2id[words[idx-2]], w2id[words[idx-1]], w2id[words[idx+1]], w2id[words[idx+2]]], w2id[words[idx]]] for idx in range(2, len(words)-3)])
		np.save('text8_cbow.npy', cbow)

	if path.exists('text8_skipgrams.npy'):
		skipgrams = np.load('text8_skipgrams.npy', allow_pickle=True)
	else:
		for idx in range(2, len(words) - 3):
			skipgrams += [[w2id[words[idx]], w2id[words[idx-2]]], [w2id[words[idx]], w2id[words[idx-1]]], [w2id[words[idx]], w2id[words[idx+1]]], [w2id[words[idx]], w2id[words[idx+2]]]]
		skipgrams = np.array(skipgrams)
		np.save('text8_skipgrams.npy', skipgrams)
	
	return cbow, skipgrams

cbow, skipgrams = load_data(words)
print (cbow[0], id2w[cbow[0][0][0]], id2w[cbow[0][0][1]], id2w[cbow[0][0][2]], id2w[cbow[0][0][3]], id2w[cbow[0][1]])
print (id2w[skipgrams[0][0]], id2w[skipgrams[0][1]], id2w[skipgrams[1][0]], id2w[skipgrams[1][1]], id2w[skipgrams[2][0]], id2w[skipgrams[2][1]], id2w[skipgrams[3][0]], id2w[skipgrams[3][1]])

#Load and preprocess the data
# dataframe = pd.read_csv('airline-passengers.csv',usecols=[1],header=0)
# dataset = dataframe.values.astype(np.float32)
# dataset = dataset.astype(np.float32)

#plt.plot(dataset,label='Original Data')
#plt.legend()
#plt.show()

#Normalize the dataset
normalized_dataset = skpp.MinMaxScaler(feature_range=(0, 1)).fit_transform(dataset)

#Split into the train and test data
dataset_split = int(0.7 * len(normalized_dataset))
x_train = normalized_dataset[:dataset_split]
x_test = normalized_dataset[dataset_split:]
x_test_plot = np.full_like(normalized_dataset, np.nan)
for i in range(dataset_split, len(normalized_dataset)):
	x_test_plot[i] = normalized_dataset[i]

# plt.plot(x_train, label='train data')
# plt.plot(x_test_plot, label='test data')
# plt.legend()
#plt.show()

def make_supervised_data(dataset, n_x, n_y):	
	x = np.array([dataset[i:i+n_x] for i in range(0, len(dataset) - n_x - n_y)])
	y = np.array([dataset[i:i+n_y] for i in range(n_x, len(dataset) - n_y)])
	x = tf.data.Dataset.from_tensor_slices(x)
	y = tf.data.Dataset.from_tensor_slices(y)
	return tf.data.Dataset.zip((x, y))

#Set the parameters of the model and learning protocol
n_x = 1
n_y = 1
n_x_vars = 1
n_y_vars = 1
state_size = 8
n_timesteps = n_x
learning_rate = 0.1
epochs = 200
batch_size = 256

train = make_supervised_data(x_train, 1, 1)
test = make_supervised_data(x_test, 1, 1)
train = train.batch(batch_size)
test = test.batch(batch_size)


class Model():
	def __init__(self, hidden_dims=[10]):
		super(Model, self).__init__()
		self.layers = []
		self.trainable_params = []
		for dim in hidden_dims:
			self.layers.append(tf.keras.layers.SimpleRNN(dim, activation='tanh'))
		self.out = tf.keras.layers.Dense(1, activation='linear')
				
	def call(self, x):
		"""Run forward inference on the model. """
		for layer in self.layers:
			x = layer(x)
		x = self.out(x)
		return x

	def compute_trainable_params(self):
		ret_val = []
		for layer in self.layers:
			ret_val += layer.trainable_weights
		ret_val += self.out.trainable_weights
		return ret_val

	def get_trainable_params(self):
		return self.trainable_params

# series_predictor = Model(hidden_dims=[state_size])
# optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

# for epoch in range(epochs):
# 	print ("Epoch {}:".format(epoch + 1))
# 	avg_loss = 0
# 	batch_num = 0	
# 	for sample in train:
# 		with tf.GradientTape(persistent=True) as tape:
# 			x, y = sample[0], sample[1]
# 			y = tf.reshape(y, shape=(y.shape[0], y.shape[1]))
# 			tape.watch(x)
# 			y_hat = series_predictor.call(x)
# 			loss = tf.reduce_mean(tf.square(y - y_hat))
# 			trainable_weights = series_predictor.compute_trainable_params()

# 		avg_loss += loss					
# 		batch_num += 1
# 		grads = tape.gradient(loss, trainable_weights)
# 		optimizer.apply_gradients(zip(grads, trainable_weights))

# 	avg_loss /= batch_num
# 	print ("Average batch loss: {:.7f}".format(avg_loss))

# #Plotting the predictions and true values for the testing data
# for batch in test:
# 	x, y = batch[0], batch[1]
# 	y_hat = series_predictor.call(x)

# plt.plot(tf.reshape(y, shape=[y.shape[0]]), label='Original Data')
# plt.plot(tf.reshape(y_hat, shape=[y_hat.shape[0]]), label='Predicted Data')
# plt.legend()
# plt.show()