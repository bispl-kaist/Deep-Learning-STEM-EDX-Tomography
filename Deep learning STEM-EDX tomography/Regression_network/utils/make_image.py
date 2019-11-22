import numpy as np
import scipy.misc
import os, os.path

def saveImage(filename, output, label, input, display_img_no = 10, _cmin = 0, _cmax = 255, imageSize = 60, nc = 1, batchSize = 10):
    
    if nc == 1:
        output = np.transpose(output, (0, 2, 3, 1))*255.0
        output = np.squeeze(output, axis=3)
        input = np.transpose(input, (0, 2, 3, 1))*255.0
        input = input[:,:,:,0]
        label = np.transpose(label, (0, 2, 3, 1))*255.0
        label = label[:,:,:,0]

        diff_img = np.absolute(np.subtract(output, label)) * 5

        image = np.concatenate((input, output, label, diff_img), axis = 1)
        image = np.concatenate([image[i,:,:] for i in range(display_img_no)], axis = 1)
    else:
        raise(Exception())

    filenames = os.path.join(filename)
    scipy.misc.toimage(image, cmin = _cmin, cmax = _cmax).save(filenames)
