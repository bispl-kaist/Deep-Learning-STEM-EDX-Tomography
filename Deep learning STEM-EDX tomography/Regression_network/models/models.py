import os, sys
sys.path.append(os.getcwd())
import numpy as np
import tensorflow as tf
import tflib as lib
import tflib.ops.conv2d
import tflib.ops.deconv2d
import tflib.layer as layer

def waveDec(name, noise, Filt_low, Filt_high1, Filt_high2, Filt_high3):
    with tf.name_scope(name) as scope:
        if noise is None: 
            Exception("There is no input!")
    
        inputSize = tf.shape(noise)
        noise = tf.reshape(noise, [inputSize[0]*inputSize[1], 1, inputSize[2], inputSize[3]])
        filtSize = tf.shape(Filt_low)
        outputSize = tf.div(inputSize[2], 2)

        for idx in range(2):
            Filt_low = tf.expand_dims(Filt_low, axis=idx+2)
            Filt_high1 = tf.expand_dims(Filt_high1, axis=idx+2)
            Filt_high2 = tf.expand_dims(Filt_high2, axis=idx+2)
            Filt_high3 = tf.expand_dims(Filt_high3, axis=idx+2)

        #decomposition
        dec_low = tf.nn.conv2d(input=noise, filter=Filt_low, strides=[1, 1, 2, 2], padding='SAME', data_format='NCHW')
        dec_high1 = tf.nn.conv2d(input=noise, filter=Filt_high1, strides=[1, 1, 2, 2], padding='SAME', data_format='NCHW')
        dec_high2 = tf.nn.conv2d(input=noise, filter=Filt_high2, strides=[1, 1, 2, 2], padding='SAME', data_format='NCHW')
        dec_high3 = tf.nn.conv2d(input=noise, filter=Filt_high3, strides=[1, 1, 2, 2], padding='SAME', data_format='NCHW')
	
        dec_low = tf.reshape(dec_low, [inputSize[0], inputSize[1], outputSize, outputSize])
        dec_high1 = tf.reshape(dec_high1, [inputSize[0], inputSize[1], outputSize, outputSize])
        dec_high2 = tf.reshape(dec_high2, [inputSize[0], inputSize[1], outputSize, outputSize])
        dec_high3 = tf.reshape(dec_high3, [inputSize[0], inputSize[1], outputSize, outputSize])
        
        return dec_low, dec_high1, dec_high2, dec_high3

def waveRec(name, noise, Filt):
    with tf.name_scope(name) as scope:
        if noise is None:
            Exception("There is no input!")

        inputSize = tf.shape(noise)
        filtSize = tf.shape(Filt)
        noise = tf.reshape(noise, [inputSize[0]*inputSize[1], 1, inputSize[2], inputSize[3]])

        for idx in range(2):
            Filt = tf.expand_dims(Filt, axis=idx+2)
	
        output_shape = [inputSize[0]*inputSize[1], 1, inputSize[2]*2, inputSize[3]*2]
        rec = tf.nn.conv2d_transpose(value=noise, filter=Filt, output_shape = output_shape, strides=[1, 1, 2, 2], padding='SAME', data_format='NCHW')
        rec = tf.reshape(rec, [inputSize[0], inputSize[1], inputSize[2]*2, inputSize[3]*2])
        return rec


def avgunpool(input):
    input_trans = tf.transpose(input, perm=[0,2,3,1])
    out = tf.concat([input_trans, input_trans],3)
    out = tf.concat([out, out], 2)
    size = input_trans.get_shape().as_list()
    if None not in size[1:]:
        out_size = [-1, size[1]*2, size[2]*2, size[3]]
        out = tf.reshape(out, out_size)
        out = tf.transpose(out, perm=[0, 3, 1,2])
        return out
    else:
        sizev=tf.shape(input_trans)
        ret=tf.reshape(out, tf.stack[-1,sizev[1]*2, sizev[2]*2, size[3]])
        return ret

def CBNModule(input, bn=True, nonlinearity=tf.nn.relu, he_init=True, lname=1, dimIn=64, dimOut=128, filtSize=3, name='Generator'):
    output = lib.ops.conv2d.Conv2D('{0}.{1}'.format(name, lname), dimIn, dimOut, filtSize, input, he_init=he_init) 
    if bn:
        output = layer.Batchnorm('{0}.BN{1}'.format(name, lname), [0,2,3], output)
    output = nonlinearity(output)
    return output

def DeCBNModule(input, bn=True, nonlinearity=tf.nn.relu, he_init=True, lname=1, dimIn=64, dimOut=64, filtSize=3, name='Generator', is_avgpool=False):
    if is_avgpool==True:
        output = avgunpool(input)
    else:
        output = lib.ops.deconv2d.Deconv2D('{0}.{1}'.format(name, lname), dimIn, dimOut, filtSize, input, he_init=he_init)
    if bn:
        output = layer.Batchnorm('{0}.BN{1}'.format(name, lname), [0,2,3], output)
    output = nonlinearity(output)
    return output

def avgpool(input):
    return tf.nn.avg_pool(input, ksize=[1,1,2,2], strides=[1,1,2,2], padding='SAME', data_format='NCHW')

def DecModule(input, filt_low, filt_high1, filt_high2, filt_high3, bn=True, nonlinearity=tf.nn.relu, he_init=True, lname1=1, lname2=2, lname3=3, dim=64, filtSize=3, name='Generator'):
    output_low, output_high1, output_high2, output_high3 = waveDec('{0}.Dec{1}'.format(name, lname1), input, filt_low, filt_high1, filt_high2, filt_high3)
    output = lib.ops.conv2d.Conv2D('{0}.{1}'.format(name, lname2), dim, dim*2, filtSize, output_low, he_init=he_init) 
    if bn:
        output = layer.Batchnorm('{0}.BN{1}'.format(name, lname2), [0,2,3], output)
    output = nonlinearity(output)

    output = lib.ops.conv2d.Conv2D('{0}.{1}'.format(name, lname3), dim*2, dim*2, filtSize, output, he_init=he_init) 
    if bn:
        output = layer.Batchnorm('{0}.BN{1}'.format(name, lname3), [0,2,3], output)
    output = nonlinearity(output)
    return output, output_high1, output_high2, output_high3

def RecModule(input_concat, input_low, input_high1, input_high2, input_high3, filt_low, filt_high1, filt_high2, filt_high3, bn=True, nonlinearity=tf.nn.relu, he_init=True, lname1=1, lname2=2, lname3=3, dim=64, reduce_dim=True, filtSize=3, name='Generator'):
    output_low = waveRec('{0}.Rec_low{1}'.format(name, lname1), input_low, filt_low)
    output_high1 = waveRec('{0}.Rec_high1_{1}'.format(name, lname1), input_high1, filt_high1)
    output_high2 = waveRec('{0}.Rec_high2_{1}'.format(name, lname1), input_high2, filt_high2)
    output_high3 = waveRec('{0}.Rec_high3_{1}'.format(name, lname1), input_high3, filt_high3)
    output = tf.concat([input_concat, output_high1, output_high2, output_high3, output_low], 1)
    
    output = lib.ops.conv2d.Conv2D('{0}.{1}'.format(name, lname2), dim*5, dim, filtSize, output, he_init=he_init) 
    if bn:
        output = layer.Batchnorm('{0}.BN{1}'.format(name, lname2), [0,2,3], output)
    output = nonlinearity(output)
    
    if reduce_dim:
        output = lib.ops.conv2d.Conv2D('{0}.{1}'.format(name, lname3), dim, int(dim/2), filtSize, output, he_init=he_init) 
        if bn:
            output = layer.Batchnorm('{0}.BN{1}'.format(name, lname3), [0,2,3], output)
        output = nonlinearity(output)
        return output
    else:
        output = lib.ops.conv2d.Conv2D('{0}.{1}'.format(name, lname3), dim, dim, filtSize, output, he_init=he_init) # dim -> dim
        if bn:
            output = layer.Batchnorm('{0}.BN{1}'.format(name, lname3), [0,2,3], output)
        output = nonlinearity(output)
        return output

def TightFrameUnet(n_samples, noise=None, nc=1, isize=60, nz=None, bn=True, nonlinearity=tf.nn.relu, OUTPUT_DIM=60*60*1, initialization='he', stage=4, filtSize=3, is_concat=False, name='Generator'):
    lib.ops.conv2d.set_weights_stdev(0.02)
    lib.ops.deconv2d.set_weights_stdev(0.02)

    if initialization == 'he':
        he_init = True
    else:
        he_init = False

    if noise is None:
        noise = tf.random_normal([n_samples, nc, isize, isize])
    
    # define filters
    filt_low = tf.constant(0.5, shape=[2,2])
    filt_high1 = tf.constant([0.5,0.5,-0.5,-0.5],shape=[2,2])
    val = np.sqrt(2)
    filt_high2 = tf.constant([tf.divide(1,val), tf.divide(-1, val), 0,0], shape=[2,2])
    filt_high3 = tf.constant([0, 0, tf.divide(1,val), tf.divide(-1, val)], shape=[2,2])

    # stage 0_dec
    if is_concat:
        input_nc = nc*3
    else:
        input_nc = nc

    output_feature01 = lib.ops.conv2d.Conv2D('{0}.Input'.format(name), input_nc, 64, filtSize, noise, he_init=he_init) 
    if bn:
        output_feature01 = layer.Batchnorm('{0}.BN01'.format(name), [0,2,3], output_feature01)
    output_feature01 = nonlinearity(output_feature01) 

    output_feature02 = lib.ops.conv2d.Conv2D('{0}.02'.format(name), 64, 64, filtSize, output_feature01, he_init=he_init)
    if bn:
        output_feature02 = layer.Batchnorm('{0}.BN02'.format(name), [0,2,3], output_feature02)
    output_feature02 = nonlinearity(output_feature02)

    # stage 1_dec
    output_feature1, output_dec1_high1, output_dec1_high2, output_dec1_high3 = DecModule(output_feature02, filt_low, filt_high1, filt_high2, filt_high3, lname1=1, lname2=11, lname3=12, dim=64, filtSize=filtSize, bn=bn, name=name)

    # stage 2_dec
    output_feature2, output_dec2_high1, output_dec2_high2, output_dec2_high3 = DecModule(output_feature1, filt_low, filt_high1, filt_high2, filt_high3, lname1=2, lname2=21, lname3=22, dim=128, filtSize=filtSize, bn=bn, name=name)


    # stage 3_dec
    output_feature3, output_dec3_high1, output_dec3_high2, output_dec3_high3 = DecModule(output_feature2, filt_low, filt_high1, filt_high2, filt_high3, lname1=3, lname2=31, lname3=32, dim=256, filtSize=filtSize, bn=bn, name=name)

    # stage 4_dec
    output_dec4_low, output_dec4_high1, output_dec4_high2, output_dec4_high3 = waveDec('{0}.Dec4'.format(name), output_feature3, filt_low, filt_high1, filt_high2, filt_high3) 
    output_dec4_low = lib.ops.conv2d.Conv2D('{0}.41'.format(name), 512, 1024, filtSize, output_dec4_low, he_init=he_init) 
    if bn:
        output_dec4_low = layer.Batchnorm('{0}.BN41'.format(name), [0,2,3], output_dec4_low)
    output_dec4_low = nonlinearity(output_dec4_low)
    output_dec4_low = lib.ops.conv2d.Conv2D('{0}.42'.format(name), 1024, 512, filtSize, output_dec4_low, he_init=he_init)
    if bn:
        output_dec4_low = layer.Batchnorm('{0}.BN42'.format(name), [0,2,3], output_dec4_low)
    output_dec4_low = nonlinearity(output_dec4_low)

    # stage 4_rec
    output_feature5 = RecModule(output_feature3, output_dec4_low, output_dec4_high1, output_dec4_high2, output_dec4_high3, filt_low, filt_high1, filt_high2, filt_high3, lname1=5, lname2=51, lname3=52, dim=512, filtSize=filtSize, bn=bn, name=name)

    # stage 3_rec
    output_feature6 = RecModule(output_feature2, output_feature5, output_dec3_high1, output_dec3_high2, output_dec3_high3, filt_low, filt_high1, filt_high2, filt_high3, lname1=6, lname2=61, lname3=62, dim=256, filtSize=filtSize, bn=bn, name=name)

    # stage 2_rec
    output_feature7 = RecModule(output_feature1, output_feature6, output_dec2_high1, output_dec2_high2, output_dec2_high3, filt_low, filt_high1, filt_high2, filt_high3, lname1=7, lname2=71, lname3=72, dim=128, filtSize=filtSize, bn=bn, name=name)

    # stage 1_rec
    output_feature8 = RecModule(output_feature02, output_feature7, output_dec1_high1, output_dec1_high2, output_dec1_high3, filt_low, filt_high1, filt_high2, filt_high3, lname1=8, lname2=81, lname3=82, dim=64, reduce_dim=False, filtSize=filtSize, bn=bn, name=name)

    output = lib.ops.conv2d.Conv2D('{0}.9'.format(name), 64, nc, 1, output_feature8, he_init=he_init) 

    lib.ops.conv2d.unset_weights_stdev()
    lib.ops.deconv2d.unset_weights_stdev()
    return output

def Unet(n_samples, noise=None, nc=1, isize=60, nz=None, bn=True, nonlinearity=tf.nn.relu, OUTPUT_DIM=60*60*1, initialization='he', stage=4, filtSize=3, is_concat=False, name='Generator'):
    lib.ops.conv2d.set_weights_stdev(0.02)
    lib.ops.deconv2d.set_weights_stdev(0.02)

    if initialization == 'he':
        he_init = True
    else:
        he_init = False

    if noise is None:
        noise = tf.random_normal([n_samples, nc, isize, isize])

    # stage 0_dec
    if is_concat:
        input_nc = nc*3
    else:
        input_nc = nc

    output_feature01 = lib.ops.conv2d.Conv2D('{0}.Input'.format(name), input_nc, 64, filtSize, noise, he_init=he_init) 
    if bn:
        output_feature01 = layer.Batchnorm('{0}.BN01'.format(name), [0,2,3], output_feature01)
    output_feature01 = nonlinearity(output_feature01) 

    output_feature02 = lib.ops.conv2d.Conv2D('{0}.02'.format(name), 64, 64, filtSize, output_feature01, he_init=he_init) # 64 -> 64
    if bn:
        output_feature02 = layer.Batchnorm('{0}.BN02'.format(name), [0,2,3], output_feature02)
    output_feature02 = nonlinearity(output_feature02)

    # stage 1_dec
    output_feature11 = avgpool(output_feature02)    
    output_feature12 = CBNModule(output_feature11, bn=bn, nonlinearity=nonlinearity, he_init=he_init, lname=12, dimIn=64, dimOut=128, filtSize=filtSize, name=name)	
    output_feature13 = CBNModule(output_feature12, bn=bn, nonlinearity=nonlinearity, he_init=he_init, lname=13, dimIn=128, dimOut=128, filtSize=filtSize, name=name)

    output_feature21 = avgpool(output_feature13)
    output_feature22 = CBNModule(output_feature21, bn=bn, nonlinearity=nonlinearity, he_init=he_init, lname=22, dimIn=128, dimOut=256, filtSize=filtSize, name=name)
    output_feature23 = CBNModule(output_feature22, bn=bn, nonlinearity=nonlinearity, he_init=he_init, lname=23, dimIn=256, dimOut=256, filtSize=filtSize, name=name)

    output_feature31 = avgpool(output_feature23)
    output_feature32 = CBNModule(output_feature31, bn=bn, nonlinearity=nonlinearity, he_init=he_init, lname=32, dimIn=256, dimOut=512, filtSize=filtSize, name=name)
    output_feature33 = CBNModule(output_feature32, bn=bn, nonlinearity=nonlinearity, he_init=he_init, lname=33, dimIn=512, dimOut=512, filtSize=filtSize, name=name)

    output_feature41 = avgpool(output_feature33)
    output_feature42 = CBNModule(output_feature41, bn=bn, nonlinearity=nonlinearity, he_init=he_init, lname=42, dimIn=512, dimOut=1024, filtSize=filtSize, name=name)
    output_feature43 = CBNModule(output_feature42, bn=bn, nonlinearity=nonlinearity, he_init=he_init, lname=43, dimIn=1024, dimOut=512, filtSize=filtSize, name=name)

    output_feature51 = DeCBNModule(output_feature43, bn=bn, nonlinearity=nonlinearity, he_init=he_init, lname=51, dimIn=512, dimOut=512, filtSize=filtSize, name=name, is_avgpool=False)
    output_feature51 = tf.concat([output_feature51, output_feature33], 1)
    output_feature52 = CBNModule(output_feature51, bn=bn, nonlinearity=nonlinearity, he_init=he_init, lname=52, dimIn=1024, dimOut=512, filtSize=filtSize, name=name)
    output_feature53 = CBNModule(output_feature52, bn=bn, nonlinearity=nonlinearity, he_init=he_init, lname=53, dimIn=512, dimOut=256, filtSize=filtSize, name=name)

    output_feature61 = DeCBNModule(output_feature53, bn=bn, nonlinearity=nonlinearity, he_init=he_init, lname=61, dimIn=256, dimOut=256, filtSize=filtSize, name=name, is_avgpool=False)
    output_feature61 = tf.concat([output_feature61, output_feature23], 1)
    output_feature62 = CBNModule(output_feature61, bn=bn, nonlinearity=nonlinearity, he_init=he_init, lname=62, dimIn=512, dimOut=256, filtSize=filtSize, name=name)
    output_feature63 = CBNModule(output_feature62, bn=bn, nonlinearity=nonlinearity, he_init=he_init, lname=63, dimIn=256, dimOut=128, filtSize=filtSize, name=name)

    output_feature71 = DeCBNModule(output_feature63, bn=bn, nonlinearity=nonlinearity, he_init=he_init, lname=71, dimIn=128, dimOut=128, filtSize=filtSize, name=name, is_avgpool=False)
    output_feature71 = tf.concat([output_feature71, output_feature13], 1)
    output_feature72 = CBNModule(output_feature71, bn=bn, nonlinearity=nonlinearity, he_init=he_init, lname=72, dimIn=256, dimOut=128, filtSize=filtSize, name=name)
    output_feature73 = CBNModule(output_feature72, bn=bn, nonlinearity=nonlinearity, he_init=he_init, lname=73, dimIn=128, dimOut=64, filtSize=filtSize, name=name)

    output_feature81 = DeCBNModule(output_feature73, bn=bn, nonlinearity=nonlinearity, he_init=he_init, lname=81, dimIn=64, dimOut=64, filtSize=filtSize, name=name, is_avgpool=False)
    output_feature81 = tf.concat([output_feature81, output_feature02], 1)
    output_feature82 = CBNModule(output_feature81, bn=bn, nonlinearity=nonlinearity, he_init=he_init, lname=82, dimIn=128, dimOut=64, filtSize=filtSize, name=name)
    output_feature83 = CBNModule(output_feature82, bn=bn, nonlinearity=nonlinearity, he_init=he_init, lname=83, dimIn=64, dimOut=64, filtSize=filtSize, name=name)

    output = lib.ops.conv2d.Conv2D('{0}.91'.format(name), 64, nc, 1, output_feature83, he_init=he_init) 

    lib.ops.conv2d.unset_weights_stdev()
    lib.ops.deconv2d.unset_weights_stdev()
    return output

