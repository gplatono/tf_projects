import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, SimpleRNN
from tensorflow.keras.optimizers import RMSprop, SGD

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
n_classes = 10
n_classes = 10
x_train = x_train.reshape(-1,28,28)
x_test = x_test.reshape(-1,28,28)
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# create and fit the SimpleRNN model
model = Sequential()
model.add(SimpleRNN(units=16, activation='relu', input_shape=(28,28)))
model.add(Dense(n_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.01), metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, batch_size=100, epochs=20)
score = model.evaluate(x_test, y_test)
print('\nTest loss:', score[0])
print('Test accuracy:', score[1])
