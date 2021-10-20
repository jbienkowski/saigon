import os
from .base.model_base import ModelBase
from time import strftime
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import numpy as np


class Model002(ModelBase):
    num_examples = 0
    index_in_epoch = 0

    def __init__(self, cfg, **kwargs):
        ModelBase.__init__(self, cfg=cfg, model_name="model_two", **kwargs)
        self._define_model()

    def _define_model(self):
        pass

    def setup_layer(self, input, weight_dim, bias_dim, name):
        with tf.name_scope("name"):
            initial_w = tf.truncated_normal(shape=weight_dim, stddev=0.1, seed=42)
            w = tf.Variable(initial_value=initial_w, name="W")

            initial_b = tf.constant(value=0.0, shape=bias_dim)
            b = tf.Variable(initial_value=initial_b, name="B")

            layer_in = tf.matmul(input, w) + b

            if name == "out":
                layer_out = tf.nn.softmax(layer_in)
            else:
                layer_out = tf.nn.relu(layer_in)

            tf.summary.histogram("weights", w)
            tf.summary.histogram("biases", b)

            return layer_out

    def next_batch(self, batch_size, data, labels):
        global num_examples
        global index_in_epoch

        start = self.index_in_epoch
        self.index_in_epoch += batch_size

        if self.index_in_epoch > self.num_examples:
            start = 0
            index_in_epoch = batch_size

        end = index_in_epoch

        return data[start:end], labels[start:end]

    def run(self):
        tf.compat.v1.disable_eager_execution()

        self.y_train = np.eye(2)[self.y_train]
        self.y_test = np.eye(2)[self.y_test]

        # Init TF graph
        X = tf.placeholder(tf.float32, shape=[None, self.total_inputs], name="X")
        Y = tf.placeholder(tf.float32, shape=[None, self.NR_CLASSES], name="labels")

        n_hidden1 = 512
        n_hidden2 = 64

        layer_1 = self.setup_layer(
            X,
            weight_dim=[self.total_inputs, n_hidden1],
            bias_dim=[n_hidden1],
            name="layer_1",
        )

        layer_drop = tf.nn.dropout(layer_1, rate=0.8, name="dropout_layer")

        layer_2 = self.setup_layer(
            layer_drop,
            weight_dim=[n_hidden1, n_hidden2],
            bias_dim=[n_hidden2],
            name="layer_2",
        )

        output = self.setup_layer(
            layer_2, weight_dim=[n_hidden2, 2], bias_dim=[2], name="out"
        )

        # Folder for Tensorboard

        folder_name = f"{self.model_name} at {strftime('%H:%M')}"
        directory = os.path.join("log/", folder_name)

        try:
            os.makedirs(directory)
        except OSError as exception:
            print(exception.strerror)
        else:
            print("Successfully created dirs!")

        with tf.name_scope("loss_calc"):
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=output)
            )

        with tf.name_scope("optimizer"):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            train_step = optimizer.minimize(loss)

        with tf.name_scope("accuracy_calc"):
            correct_pred = tf.equal(tf.argmax(output, axis=1), tf.argmax(Y, axis=1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        with tf.name_scope("performance"):
            tf.summary.scalar("accuracy", accuracy)
            tf.summary.scalar("cost", loss)

        # with tf.name_scope('show_image'):
        #     x_image = tf.reshape(X, [-1, 28, 28, 1])
        #     tf.summary.image('image_input', x_image, max_outputs=4)

        sess = tf.Session()

        merged_summary = tf.summary.merge_all()

        train_writer = tf.summary.FileWriter("log" + "/train")
        train_writer.add_graph(sess.graph)

        validation_writer = tf.summary.FileWriter("log" + "/validation")

        init = tf.global_variables_initializer()
        sess.run(init)

        num_examples = self.y_train.shape[0]
        nr_iterations = int(num_examples / self.samples_per_batch)

        self.index_in_epoch = 0

        for epoch in range(self.epochs):
            # ============= Training Dataset =========
            for i in range(nr_iterations):

                batch_x, batch_y = self.next_batch(
                    batch_size=self.samples_per_batch,
                    data=self.x_train,
                    labels=self.y_train,
                )

                feed_dictionary = {X: batch_x, Y: batch_y}

                sess.run(train_step, feed_dict=feed_dictionary)

            s, batch_accuracy = sess.run(
                fetches=[merged_summary, accuracy], feed_dict=feed_dictionary
            )

            train_writer.add_summary(s, epoch)

            print(f"Epoch {epoch} \t| Training Accuracy = {batch_accuracy}")

            # ================== Validation ======================

            summary = sess.run(
                fetches=merged_summary, feed_dict={X: self.x_test, Y: self.y_test}
            )
            validation_writer.add_summary(summary, epoch)

        print("Done training!")