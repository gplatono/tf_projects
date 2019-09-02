import numpy as np
import tensorflow as tf

#model params
hid_num = 10
lrate = 0.01
iter = 300

data = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0.9, 0.9], [0.1, 0], [0, 1.1], [0.9, 0], [0.8, 0.8], [0.1, 0.2]])
labels = np.array([[1, 0], [0, 1], [0, 1], [1, 0], [1,0], [1,0], [0,1], [0,1], [1,0], [1,0]])

x = tf.placeholder(tf.float32, data.shape)
y = tf.placeholder(tf.float32, labels.shape)

wh1 = tf.Variable(tf.random_normal([2, hid_num]))
wb1 = tf.Variable(tf.random_normal([hid_num]))

wo = tf.Variable(tf.random_normal([hid_num, 2]))
wbo = tf.Variable(tf.random_normal([2]))

layer1 = tf.sigmoid(tf.add(tf.matmul(x, wh1), wb1))
output = tf.add(tf.matmul(layer1, wo), wbo)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=labels))
optimizer = tf.train.AdamOptimizer(learning_rate=lrate)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(output, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()


#computation = tf.matmul(tf.constant([[2,2]]), tf.constant([[3], [3]]))
with tf.Session() as sess:
    sess.run(init)
    for step in range(iter):
        sess.run(train_op, feed_dict={x: data, y: labels})
        loss, acc = sess.run([loss_op, accuracy], feed_dict={x: data, y: labels})
        print("Step " + str(step) + ", Minibatch Loss= " + \
              "{:.4f}".format(loss) + ", Training Accuracy= " + \
              "{:.3f}".format(acc))


    #print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: data, y: labels}))
