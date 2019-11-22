import os, sys
sys.path.append(os.getcwd())
import time
import tensorflow as tf
import numpy as np
import models.models as models
import tflib as lib
import tflib.plot
import math
from datasets import loadData
import utils.make_image as makeImage
from scipy import signal
from scipy import misc
import scipy.io as sio
import timeit
from PIL import Image
import scipy

class Model_test(object):
    def __init__(self, sess, opt):
        self.sess = sess
        self.method = opt.method
        self.batchSize = opt.batchSize
        self.imageSize = opt.imageSize 
        self.filtSize = opt.filtSize
        self.nc = opt.nc
        self.dataroot_val = opt.dataroot_val
        self.cmin = opt.min_color_index
        self.cmax = opt.max_color_index
        self.display_no	= opt.no_display_img
        self.experiment	= opt.experiment
        self.optimizer = opt.optimizer
        self._lrG = opt.lrG
        self.lrG_half_life = opt.lrG_half_life
        self.lrG_lower_bound = opt.lrG_lower_bound
        self.ngpus = opt.ngpus
        self.epochs = opt.nepochs
        self.outputDim = opt.imageSize*opt.imageSize*opt.nc
        self.initialization = opt.initialization
        self.nsave = opt.nsave
        self.load_pretrained_generator_path = opt.load_pretrained_generator_path 
        self.residual = opt.residual
        self.bn = opt.bn
        self.no_random = opt.no_random

        if opt.model == 'TightFrameUnet':
            self.generator 	= models.TightFrameUnet
        elif opt.model == 'Unet':
            self.generator 	= models.Unet
        else:
            raise NameError

        self.devices = ['/gpu:{}'.format(i) for i in range(self.ngpus)]
        self._build_model(opt)

    def _build_model(self,opt):

	# input data
        self.all_input_data_conv = tf.placeholder(tf.float32, shape=[None, self.nc, None, None])
        self.split_input_data_conv = tf.split(self.all_input_data_conv, len(self.devices))
	
        self.all_fake_data = []

        for device_index, (device, input_data_conv) in enumerate(zip(self.devices, self.split_input_data_conv)):
            with tf.device(device):
                self.input_data = input_data_conv
        
                if self.residual:
                    gen_name = 'Generator_residual'
                else:		
                    gen_name = 'Generator_img'

                self.fake_data = self.generator(None, noise=input_data_conv, nc=self.nc, bn=self.bn,  
                               isize=tf.shape(input_data_conv)[2], OUTPUT_DIM=self.outputDim, initialization=self.initialization, filtSize=self.filtSize, name=gen_name)

                if self.residual:
                    self.fake_data = tf.nn.relu(self.fake_data) + input_data_conv
	
                self.all_fake_data.append(self.fake_data)

            self.all_fake_data = tf.concat(self.all_fake_data, axis=0)
            self.g_vars = lib.params_with_name(gen_name)

            max_to_keep = int(math.ceil(float(self.epochs)/max(float(self.nsave),1)))
            self.gen_saver = tf.train.Saver(var_list=self.g_vars, max_to_keep=max_to_keep)

    def Test(self, opt):
        self.lrG = tf.placeholder(tf.float32)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.load(gen=True)

        test_idx = loadData.get_data_idx(root=self.dataroot_val, no_random=self.no_random)
        saveDir = os.path.join(self.dataroot_val, 'Result')
        os.system('mkdir {0}'.format(saveDir))
        saveDir_sub = os.path.join(saveDir, 'SAIT')
        os.system('mkdir {0}'.format(saveDir_sub))
	
        data_iteration	= -1
        testDataNum = len(test_idx)

        while data_iteration < testDataNum-1:
            data_iteration += 1
            lib.plot.tick()

            self.path_img = test_idx[data_iteration][0]
            overallImg = self.pil_loader(self.path_img)
            overallImg = np.asarray(overallImg, dtype=np.float32)
            self.path_label = test_idx[data_iteration][1]
            overallLabel = np.asarray(self.pil_loader(self.path_label), dtype=np.float32)

            maxVal = np.amax(overallLabel)
            overallInput = overallImg/(maxVal+1e-15)
            _, ImgW = overallInput.shape

            self.start = timeit.default_timer()

            _input = np.expand_dims(np.expand_dims(overallInput, axis=0), axis=0)

            _fake_data = self.sess.run(self.all_fake_data, feed_dict={self.all_input_data_conv:_input})

            makeImage.saveImage('{0}/during_training.png'.format(self.experiment), output=_fake_data*255.0, label=_input*255.0, input=_input*255.0, 
				display_img_no=self.display_no, _cmin=self.cmin, _cmax=self.cmax, imageSize=ImgW, nc=self.nc, batchSize=1)

            self.stop = timeit.default_timer()

            print (self.stop-self.start)
	    
            _rec = _fake_data[0,0,:,:]*(maxVal+1e-15)

            path_rec = self.path_img.replace('Input', 'Result')
            sio.savemat('{0}'.format(path_rec), mdict={'rec':_rec})
		
    def save(self, step, gen=False):
        if gen:
            save_path 	= self.gen_saver.save(self.sess, "{0}/gen".format(self.experiment), global_step=step)
            print('Generator model saved in the file : %s' %save_path)

    def load(self, gen=False):
        if gen:
            self.gen_saver.restore(self.sess, self.load_pretrained_generator_path)
            print('Restore model')

    def pil_loader(self, path):
        target 	= sio.loadmat(path)['edx_img']
        return target


	
