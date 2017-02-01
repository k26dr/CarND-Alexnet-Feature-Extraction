import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
import numpy as np

# Config
LEARNING_RATE = .001
EPOCHS = 10
BATCH_SIZE = 256

# TODO: Load traffic signs data.
with open('train.p', 'rb') as f:
    data = pickle.load(f)

# TODO: Split data into training and validation sets.
X_train, X_val, y_train, y_val = train_test_split(data['features'], data['labels'], test_size=0.2)

# TODO: Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
resized = tf.image.resize_images(x, (227, 227))


# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
weights = tf.Variable(tf.truncated_normal((4096, 43), stddev=0.01))
biases = tf.Variable(tf.zeros(43))
logits = tf.nn.xw_plus_b(fc7, weights, biases)

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
one_hot_y = tf.one_hot(y, 43)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# TODO: Train and evaluate the feature extraction model.
n_train = len(X_train)
n_val = len(X_val)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(EPOCHS):
        print("EPOCH {} ...".format(i+1))
        for offset in range(0, n_train, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={ x: batch_x, y: batch_y })
        val_accuracies = []
        for offset in range(0, n_val, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_val[offset:end], y_val[offset:end]
            batch_accuracy = sess.run(accuracy_operation, feed_dict={ x: batch_x, y: batch_y })
            val_accuracies.append(batch_accuracy)
        print("Validation Accuracy = {:.3f}".format(np.mean(val_accuracies)))
        print()
