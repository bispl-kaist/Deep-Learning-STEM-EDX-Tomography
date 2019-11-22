import os, sys
sys.path.append(os.getcwd())

import tensorflow as tf
import argparse
from train import Model
from train_concat import Model_concat
from test_all import Model_test
from test_all_concat import Model_test_concat

main_parser = argparse.ArgumentParser(add_help=False)

main_parser.add_argument('--method', default='regression')
main_parser.add_argument('--phase', default='train', help='train | test')
main_parser.add_argument('--dataroot', default='', help='path to dataset')
main_parser.add_argument('--dataroot-val', default='', help='path to dataset for validation')
main_parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
main_parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
main_parser.add_argument('--imageSize', type=int, default=64, help='the heihgt of the input image to network')
main_parser.add_argument('--no-random', type=int, default=10, help='the number of bootstrap subsampling')
main_parser.add_argument('--nc', type=int, default=3,   help='number of channels in input (image)')
main_parser.add_argument('--experiment', default=None, help='Where to store samples and models')
main_parser.add_argument('--nsave', type=int, default=0, help='number of (generator) iterations to save models')
main_parser.add_argument('--model', default='TightFrameUnet', help='model: Unet | TightFrameUnet')
main_parser.add_argument('--lrG', type=float, default=0.00005, help='learning rate for network, default=0.00005')
main_parser.add_argument('--lrG-lower-bound', type=float, default=0.000005, help='lower bound of learning rate for network')
main_parser.add_argument('--lrG-half-life', type=int, default=2000, help='number of iterations until learning rate is halved')
main_parser.add_argument('--load-pretrained-generator-path', default='', help='path to saved variables for generator')
main_parser.add_argument('--filtSize', type=int, default=3)
main_parser.add_argument('--ngpus', type=int, default=1)
main_parser.add_argument('--optimizer', default='rmsprop', help='rmsprop | adam')
main_parser.add_argument('--nepochs', type=int, default=200000)
main_parser.add_argument('--regcost', default='l2', help ='l1| l2')

# store figure setting
main_parser.add_argument('--max-color-index', type=float, default=1.0, help='max value of colorbar for saving results')
main_parser.add_argument('--min-color-index', type=float, default=0.0, help='min value of colorbar for saving results')
main_parser.add_argument('--no-display-img', type=int, default=4, help='no of images to display results')

# arguments for residual learning
main_parser.add_argument('--residual',
    dest='residual', action='store_true', help='flag for residual learning')
main_parser.add_argument('--no-residual',
    dest='residual', action='store_false', help='flag for residual learning')
main_parser.set_defaults(residual=True)

# arguments for bn
main_parser.add_argument('--bn',
    dest='bn', action='store_true', help='flag for bn')
main_parser.add_argument('--no-bn',
    dest='bn', action='store_false', help='flag for bn')
main_parser.set_defaults(bn=False)

# arguments for si dataset using concat
main_parser.add_argument('--concat',
        dest='concat', action='store_true', help='flag for concat')
main_parser.add_argument('--no-concat',
        dest='concat', action='store_false', help='flag for concat')
main_parser.set_defaults(concat=False)

# parse arguments
opt = main_parser.parse_args()

# print options

os.system('mkdir samples')
if opt.experiment is None:
    opt.experiment = 'samples/experiment'
os.system('mkdir {0}'.format(opt.experiment))

opt.initialization = 'normal'

tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
tfconfig.allow_soft_placement = True

with tf.Session(config=tfconfig) as sess:
    if opt.phase == 'train':
        if opt.concat:
            model = Model_concat(sess,opt)
        else:
            model = Model(sess, opt)
        model.Train(opt)

    elif opt.phase == 'test':
        if opt.concat:
            model = Model_test_concat(sess,opt)
        else:
            model = Model_test(sess,opt)
        model.Test(opt)
