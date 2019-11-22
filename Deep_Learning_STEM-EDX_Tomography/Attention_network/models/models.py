import os, sys
sys.path.append(os.getcwd())
import numpy as np
import tensorflow as tf

import tflib as lib
import tflib.ops.linear
import tflib.layer as layer

def attention(n_samples, noise=None, nc=1, nonlinearity=layer.LeakyReLU, isize=60, name='attention', is_avg=False):
    lib.ops.linear.set_weights_stdev(0.02)

    initial = 'normal'

    attention_weight = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(noise, axis=3), axis=2), axis=1)
    attetion_weight = tf.expand_dims(attention_weight, axis=1)

    attention_weight = tf.reshape(attention_weight, [-1, n_samples])

    attention_weight = lib.ops.linear.Linear('{0}.attention1'.format(name), n_samples, 128, attention_weight, initialization=initial)
    attention_weight = lib.ops.linear.Linear('{0}.attention2'.format(name), 128, 128, attention_weight, initialization=initial)
    attention_weight = lib.ops.linear.Linear('{0}.attention3'.format(name), 128, n_samples, attention_weight, initialization=initial)


    attention_weight = tf.squeeze(attention_weight)
    attention_weight = tf.expand_dims(tf.expand_dims(tf.expand_dims(attention_weight, 1),2),3)

    attention_rep = tf.tile(attention_weight, [1, nc, isize, isize])

    output = tf.expand_dims(tf.reduce_sum(tf.multiply(attention_rep, noise),0),0)
    if is_avg:
        avg_filter = tf.ones([3, 3, tf.shape(output)[1], tf.shape(output)[1]], dtype=tf.float32)
        avg_filter = tf.divide(avg_filter, 9)
        output = tf.nn.conv2d(output, avg_filter, strides=[1,1,1,1], padding='SAME', data_format='NCHW')


    return output



