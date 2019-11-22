import numpy as np
from PIL import Image
import os
import os.path
import random
import scipy.io as sio


class Batch(object):
    def __init__(self, opt):
        self.imageSize = opt.imageSize
        self.nc = opt.nc
        self.batchSize = opt.batchSize


    def getBatch(self, data_index, nbatch):
        labels = np.zeros((self.batchSize, self.imageSize, self.imageSize, self.nc), dtype=np.float32)
        inputs = np.zeros_like(labels)
        concat_inputs = np.zeros((self.batchSize, self.imageSize, self.imageSize, self.nc*2), dtype=np.float32)
        index = np.arange(nbatch*self.batchSize, (nbatch+1)*self.batchSize)

        nCount 	= -1	
        for idx in index:
            nCount = nCount + 1
            self.path = data_index[idx]
            self.path_label = self.path[2]
            self.path_concat_input = self.path[1]
            self.path_input = self.path[0]
            self.label, self.concat_input, self.input = self.pil_loader()
            self.label = np.asarray(self.label, dtype=np.float32)

	    # data augmentation 
            randVal_flip = np.random.rand(1)
            if randVal_flip>0.6:
                self.label = np.flip(self.label,1)
                self.concat_input = np.flip(self.concat_input, 1)
                self.input = np.flip(self.input, 1)	    
            elif randVal_flip<0.3:
                self.label = np.flip(self.label,0)
                self.concat_input = np.flip(self.concat_input,0)
                self.input = np.flip(self.input, 0)

            randVal_rot	= np.random.rand(1)
            if randVal_rot>0.7:
                self.label = np.rot90(self.label)
                self.concat_input = np.rot90(self.concat_input)
                self.input = np.rot90(self.input)

            h, w = self.label.shape

            if h < self.imageSize :
                resx = self.imageSize-h
                y1 = random.randint(0, resx)
                self.label = np.lib.pad(self.label, ((y1, resx-y1), (0,0)), 'symmetric')
                self.concat_input = np.lib.pad(self.concat_input, ((y1, resx-y1), (0,0)), 'symmetric')
                self.input = np.lib.pad(self.input, ((y1, resx-y1), (0,0)), 'symmetric')
                h, _ = self.label.shape
	
            if w < self.imageSize:
                resy = self.imageSize-w
                x1 = random.randint(0, resy)
                self.label = np.lib.pad(self.label, ((0,0), (x1, resy-x1)), 'symmetric')
                self.concat_input = np.lib.pad(self.concat_input, ((0,0), (x1, resy-x1)), 'symmetric')
                self.input = np.lib.pad(self.input, ((0,0), (x1, resy-x1)), 'symmetric')
                _, w = self.label.shape
		
            if (h == self.imageSize) and (w == self.imageSize):
                labelCrop = self.label
                concatinputCrop = self.concat_input
                inputCrop = self.input
	
            else:
                x1 = random.randint(0, w - self.imageSize)
                y1 = random.randint(0, h - self.imageSize)    
                labelCrop = self.label[ y1 : y1+self.imageSize, x1 : x1+self.imageSize]
                concatinputCrop = self.concat_input[ y1 : y1+self.imageSize, x1 : x1+self.imageSize]
                inputCrop = self.input[ y1 : y1+self.imageSize, x1 : x1+self.imageSize]

            maxVal = np.amax(labelCrop)
            inputCrop = inputCrop/(maxVal+1e-15)
            labelCrop = labelCrop/(maxVal+1e-15)
            concatinputCrop = concatinputCrop/(maxVal+1e-15)

            labels[nCount,:,:,0] = labelCrop	
            concat_inputs[nCount,:,:,:] = concatinputCrop
            inputs[nCount,:,:,0] = inputCrop
			   	
        labels = np.transpose(labels, (0, 3, 1, 2))
        concat_inputs = np.transpose(concat_inputs, (0, 3, 1, 2))
        inputs = np.transpose(inputs, (0, 3, 1, 2)) 	
          	 
        return inputs, concat_inputs, labels
    
    
    def pil_loader(self):
        label_img 	    = sio.loadmat(self.path_label)['edx_img']
        concat_input_img    = sio.loadmat(self.path_concat_input)['edx_img']
        input_img 	    = sio.loadmat(self.path_input)['edx_img']

        return label_img, concat_input_img, input_img
    

