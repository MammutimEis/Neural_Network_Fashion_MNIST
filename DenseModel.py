import tensorflow as tf


class DenseModel:
    def __init__(self, config):
        self.config = config
        self.build_model()

    def build_model(self):
        self.x = tf.placeholder(tf.float32, shape=[None, 28 * 28])
        self.y = tf.placeholder(tf.uint8, shape=[None])
        labels_onehot = tf.one_hot(indices=self.y, depth=self.config['n_classes'])

        weight_initer = tf.truncated_normal_initializer(mean=0.0, stddev=0.5)

        hidden = tf.layers.dense(self.x, units=self.config['n_hidden'], activation=tf.nn.relu,
                                 kernel_initializer=weight_initer)
        output = tf.layers.dense(hidden, units=self.config['n_classes'], activation=tf.nn.relu,
                                 kernel_initializer=weight_initer)
        self.loss = tf.losses.softmax_cross_entropy(onehot_labels=labels_onehot, logits=output)

        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config['learning_rate'])
        self.train = optimizer.minimize(self.loss)

        # evaluation measures
        correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(labels_onehot, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
