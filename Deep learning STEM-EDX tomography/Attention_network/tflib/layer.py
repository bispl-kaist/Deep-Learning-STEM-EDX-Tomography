import os, sys
sys.path.append(os.getcwd())
import numpy as np
import tensorflow as tf

import tflib as lib
import tflib.ops.linear
import tflib.ops.batchnorm
import tflib.ops.layernorm


def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs, initialization='he'):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs, initialization=initialization)
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs, initialization='he'):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs, initialization=initialization)
    return LeakyReLU(output)

def Batchnorm(name, axes, inputs, mode=''):
    if ('Discriminator' in name) and (('gp' in mode) or ('gnr' in mode)):
        if axes != [0,2,3]:
            raise(Exception('Layernorm over non-standard axes is unsupported'))
        return lib.ops.layernorm.Layernorm(name,[1,2,3],inputs)
    else:
        return lib.ops.batchnorm.Batchnorm(name,axes,inputs,fused=True)

def pixcnn_gated_nonlinearity(a, b):
    return tf.sigmoid(a) * tf.tanh(b)

