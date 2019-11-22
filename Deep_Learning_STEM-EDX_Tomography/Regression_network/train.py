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
from datasets.getBatch import Batch
import random
import utils.make_image as makeImage
from scipy import signal
import scipy.io as sio

class Model(object):
    def __init__(self, sess, opt):
        self.sess = sess
        self.method = opt.method
        self.batchSize = opt.batchSize
        self.imageSize = opt.imageSize
        self.filtSize = opt.filtSize
        self.nc = opt.nc
        self.dataroot = opt.dataroot 
        self.dataroot_val = opt.dataroot_val
        self.cmin = opt.min_color_index
        self.cmax = opt.max_color_index
        self.display_no = opt.no_display_img
        self.experiment = opt.experiment
        self.optimizer = opt.optimizer
        self._lrG = opt.lrG
        self.lrG_half_life = opt.lrG_half_life
        self.lrG_lower_bound = opt.lrG_lower_bound
        self.ngpus = opt.ngpus
        self.epochs = opt.nepochs
        self.outputDim = opt.imageSize*opt.imageSize*opt.nc
        self.initialization = opt.initialization
        self.regcost = opt.regcost
        self.nsave = opt.nsave
        self.load_pretrained_generator_path = opt.load_pretrained_generator_path 
        self.residual = opt.residual
        self.bn = opt.bn
        self.no_random = opt.no_random

        if opt.model == 'TightFrameUnet':
            self.generator = models.TightFrameUnet
        elif opt.model == 'Unet':
            self.generator = models.Unet
        else:
            raise NameError
        
        self.devices = ['/gpu:{}'.format(i) for i in range(self.ngpus)]
        self._build_model(opt)

    def _build_model(self,opt):

	# label data
        self.all_label_data_conv = tf.placeholder(tf.float32, shape=[self.batchSize, self.nc, self.imageSize, self.imageSize])
        self.split_label_data_conv = tf.split(self.all_label_data_conv, len(self.devices))
	# input data
        self.all_input_data_conv = tf.placeholder(tf.float32, shape=[self.batchSize, self.nc, self.imageSize, self.imageSize])
        self.split_input_data_conv = tf.split(self.all_input_data_conv, len(self.devices))

	# cost
        self.gen_costs = []
        self.all_fake_data = []
        
        for device_index, (device, label_data_conv, input_data_conv) in enumerate(zip(self.devices, self.split_label_data_conv, self.split_input_data_conv)):
            with tf.device(device):
                self.label_data = label_data_conv
                self.input_data = input_data_conv

                if self.residual:
                    gen_name = 'Generator_residual'
                else:		
                    gen_name = 'Generator_img'
                    
                self.fake_data = self.generator(self.batchSize/len(self.devices), noise=input_data_conv, nc=self.nc, bn=self.bn,  
					isize=self.imageSize, OUTPUT_DIM=self.outputDim, initialization=self.initialization, filtSize=self.filtSize, name=gen_name)
		
                if self.residual:
                    self.fake_data = tf.nn.relu(self.fake_data) + input_data_conv

                self.all_fake_data.append(self.fake_data)
                
                opt, gen_cost = self.calculate_cost(opt)

                self.gen_costs.append(gen_cost)
                
        self.gen_cost = tf.add_n(self.gen_costs) / len(self.devices)

        self.all_fake_data = tf.concat(self.all_fake_data, axis=0)
        self.g_vars = lib.params_with_name(gen_name)
        
        max_to_keep = int(math.ceil(float(self.epochs)/max(float(self.nsave),1)))
        self.gen_saver = tf.train.Saver(var_list=self.g_vars, max_to_keep=max_to_keep)
    
    def Train(self, opt):
        self.lrG = tf.placeholder(tf.float32)
        self.optimization()	
        
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        batchLoader = Batch(opt)

        if opt.load_pretrained_generator_path != '':
            self.load(gen=True)

	# Train loop 
        iteration = -1
        train_idx = loadData.get_data_idx(root=self.dataroot, no_random=self.no_random)
        test_idx = loadData.get_data_idx(root=self.dataroot_val, no_random=self.no_random)
        
        _fixed_test_inputs, _fixed_test_labels = batchLoader.getBatch(test_idx, 0)
        
        trainDataNum = len(train_idx)
        TotalBATCHNum = int(trainDataNum / self.batchSize)
        
        gen_iters = 1
        
        train_gen_cost = []
        val_gen_cost = []
        
        while iteration < self.epochs:	
            lib.plot.tick()
            iteration += 1
            nbatch = -1
            
            gen_costs =[]
            
            random.shuffle(train_idx)
            for i in range(TotalBATCHNum):
                if (nbatch+1) % TotalBATCHNum == 0:
                    nbatch = -1
                for i_gen in range(gen_iters):
                    nbatch += 1
                    _input, _label = batchLoader.getBatch(train_idx, nbatch)
                    
                    _gen_cost, _fake_data, _ = self.sess.run([self.gen_cost, self.all_fake_data, self.gen_train_op], 
						feed_dict={self.all_label_data_conv:_label, self.all_input_data_conv:_input, self.lrG:self._lrG})

                    gen_costs.append(_gen_cost)
                    makeImage.saveImage('{0}/during_training.png'.format(self.experiment), output=_fake_data, label=_label, input=_input,
						display_img_no=self.display_no, _cmin=self.cmin, _cmax=self.cmax, nc=self.nc, batchSize=self.batchSize)
                    makeImage.saveImage('{0}/during_training_residual.png'.format(self.experiment), output=_fake_data-_input, label=_label-_input, input=_input, 
						display_img_no=self.display_no, _cmin=self.cmin, _cmax=self.cmax, nc=self.nc, batchSize=self.batchSize)
                    
                if i %5 == 0:
                    print('Epoch:%d, process:%d/%d -- cost:%3.4f' %(iteration+1, i+1, TotalBATCHNum, _gen_cost))

	    # Display
            lib.plot.plot('train gen cost', np.mean(gen_costs))
           
	    # Updating learning rate
            if (iteration % self.lrG_half_life == self.lrG_half_life-1) and (self._lrG > self.lrG_lower_bound):
                self._lrG *=0.5
                print ('Learning rate is [%3.6f]' %(self._lrG))	

	    # validataion
            dev_gen_costs = []
            
            random.shuffle(test_idx)
            dev_TotalBATCHNum = int(math.floor(len(test_idx)/self.batchSize))	
            dev_nbatch = -1
            
            for dev_iter in range(dev_TotalBATCHNum):
                if (dev_nbatch+1) % dev_TotalBATCHNum :		
                    dev_nbatch = -1
                
                for dev_i_gen in range(gen_iters):	   
                    dev_nbatch += 1
                    _dev_input, _dev_label = batchLoader.getBatch(test_idx, dev_nbatch) 
                    
                    _dev_gen_cost, _dev_fake_data 	= self.sess.run([self.gen_cost, self.all_fake_data],
                                                   feed_dict={self.all_label_data_conv: _dev_label, self.all_input_data_conv:_dev_input})
                    
                    makeImage.saveImage('{0}/during_validation.png'.format(self.experiment), output = _dev_fake_data, label= _dev_label, input=_dev_input, 
							display_img_no = self.display_no, _cmin = self.cmin, _cmax = self.cmax, nc = self.nc, batchSize = self.batchSize)
                    
                    makeImage.saveImage('{0}/during_validation_residual.png'.format(self.experiment), output=_dev_fake_data-_dev_input, label=_dev_label-_dev_input, input=_dev_input, 
							display_img_no = self.display_no, _cmin = self.cmin, _cmax = self.cmax, nc=self.nc, batchSize=self.batchSize)
                    
                    dev_gen_costs.append(_dev_gen_cost)
                    
            lib.plot.plot('dev gen cost', np.mean(dev_gen_costs))
            
            if iteration % self.nsave == self.nsave-1:
                
                _fixed_fake_data = self.sess.run(self.all_fake_data, feed_dict={self.all_label_data_conv: _fixed_test_labels, self.all_input_data_conv: _fixed_test_inputs})
                
                makeImage.saveImage('{0}/Result_iter_{1}.png'.format(self.experiment, iteration), output=_fixed_fake_data, label=_fixed_test_labels, input=_fixed_test_inputs, 
					display_img_no = self.display_no, _cmin = self.cmin, _cmax = self.cmax, nc = self.nc, batchSize = self.batchSize)
                
            lib.plot.flush(self.experiment)	
            
            if (self.nsave>0) and (iteration % self.nsave == self.nsave-1):
                _gen = True
                self.save(iteration, gen=_gen)
    
    def calculate_cost(self, opt):
        gen_cost = 0
        if self.regcost=='l2':
            gen_cost += tf.reduce_mean((self.label_data - self.fake_data)**2)
        elif self.regcost=='l1':
            gen_cost += tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.abs(self.label_data - self.fake_data),2),2))
        return opt, gen_cost

    def optimization(self):
        if self.optimizer == 'rmsprop':
            self.gen_train_op = tf.train.RMSPropOptimizer(learninig_rate=self.lrG).minimize(self.gen_cost,
                                        var_list=self.g_vars, colocate_gradients_with_ops=True)

        elif self.optimizer == 'adam':
            self.gen_train_op = tf.train.AdamOptimizer(learning_rate=self.lrG, beta1=0.9, beta2=0.999).minimize(self.gen_cost,
                                        var_list=self.g_vars, colocate_gradients_with_ops=True)

    def save(self, step, gen=False):
        if gen:
            save_path = self.gen_saver.save(self.sess, "{0}/gen".format(self.experiment), global_step=step)
            print('Generator model saved in the file : %s' %save_path)


    def load(self, gen=False, disc=False):
        if gen:
            self.gen_saver.restore(self.sess, self.load_pretrained_generator_path)
            print('Restore model')
	



