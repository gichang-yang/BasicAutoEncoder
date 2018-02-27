import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np

class AutoEncoder:
    def __init__(self,params):
        self.params = params


    def model(self,X):
        with tf.variable_scope('encoder'):
            #encoder_x = tf.reshape(X,[1,-1])
            encoder_x = X
            encoder_w = tf.get_variable(
                "w",
                shape=[encoder_x.shape[1], self.params['labels']],
                dtype=tf.float32,
                initializer= layers.xavier_initializer(),
            )
            encoder_b = tf.Variable(tf.random_normal([self.params['labels']]))
            affined_encoder = tf.matmul(encoder_x,encoder_w) + encoder_b
            lay_out = tf.tanh(affined_encoder)
            self.Z = lay_out

        with tf.variable_scope('decoder'):
            decoder_w = tf.get_variable(
                "decoder_W",
                shape=[encoder_w.shape[1],encoder_w.shape[0]],
                dtype=tf.float32,
                initializer=layers.xavier_initializer()
            )
            decoder_b = tf.Variable(tf.random_normal(encoder_x.shape))
            affined_decoder = tf.matmul(lay_out, decoder_w) + decoder_b
            out = tf.tanh(affined_decoder)
            self.out = out
            loss = tf.reduce_mean((out - X) ** 2)
            self.loss = loss
            optim = tf.train.GradientDescentOptimizer(learning_rate=self.params['learning_rate']).minimize(loss)
            self.optim = optim
        return
