import tensorflow as tf


class CnnModel:
    def __init__(self, config):
        self.config = config
        self.build_model()

    def build_model(self):
        self.x = tf.placeholder(tf.float32, shape=[None, 28 * 28])
        self.y = tf.placeholder(tf.uint8, shape=[None])
        labels_onehot = tf.one_hot(indices=self.y, depth=self.config['n_classes'])

        weight_initer = tf.truncated_normal_initializer(mean=0.0, stddev=0.5)

        # Input Layer, reshape in proper format: [batchsize, height, width, channels]
        input_layer = tf.reshape(self.x, [-1, 28, 28, 1])

        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=self.config['filters'],
            kernel_size=self.config['kernel_size'],
            padding="valid",
            activation=tf.nn.relu,
            kernel_initializer = weight_initer
        )

        # Dense Layer
        #dense = tf.layers.dense(inputs=pool2_flat, units=self.config['n_hidden'], activation=tf.nn.relu,
        #                        kernel_initializer=weight_initer)


        # output Layer
        output = tf.layers.dense(inputs=tf.contrib.layers.flatten(conv1), units=self.config['n_classes'],
                                 kernel_initializer=weight_initer)

        self.loss = tf.losses.softmax_cross_entropy(onehot_labels=labels_onehot, logits=output)

        optimizer = tf.train.AdamOptimizer()
        self.train = optimizer.minimize(self.loss)

        # evaluation measures
        correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(labels_onehot, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


