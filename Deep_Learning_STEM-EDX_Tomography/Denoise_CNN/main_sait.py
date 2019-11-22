import argparse, os

from utils      import Logger
from utils      import Parser

from models     import Models
from database   import Database
from train      import Trainer

parser = argparse.ArgumentParser(description='initial params for training the network')
parser.add_argument('--mode', default='test', choices=["train", "test"], help='mode : [train] and [test]')
parser.add_argument('--train_continue', default=False, help='if true, continue the network training using saved network')

parser.add_argument('--scope', default='sait_cgan_avg3_s', help='add the scope at the checkpoint and log')
parser.add_argument('--input_dir', default='./data/sait_s/avg3', help='directory where training datasets are located')
parser.add_argument('--output_dir', default='./test/sait/s/avg3', help='directory where test results are saved')
parser.add_argument('--checkpoint_dir', default='./checkpoint', help='directory where saving the trained network')
parser.add_argument('--log_dir', default='./log', help='directory where log files are located')

parser.add_argument('--checkpoint_id', type=int, default=-1, help='directory where saving the trained network')

parser.add_argument('--network_type', default='cgan', help='network type : [unet], [resnet], [autoencoder], and etc.')
parser.add_argument('--learning_type', default='standard', choices=["standard", "residual"],  help='learning type : [standard] and [residual]')

parser.add_argument('--epoch_num', type=int, default=30, help='the number of epoch')
parser.add_argument('--batch_size', type=int, default=4, help='mini-batch size')
parser.add_argument('--initializer', default='xavier', help='initializer such as [he], [gaussian], [normal] and [xavier].')
parser.add_argument('--regularizer', default='l2', help='regularizer such as [l1] and [l2].')
parser.add_argument('--optimizer', default='adam', help='optimizer : [sgd], [adam], and [rmsprop].')
parser.add_argument('--loss', default='l2', help='loss : [l1], [l2], [log], and [softmax].')

parser.add_argument('--beta', type=float, default=0.9, help='momentum term of adam.')

parser.add_argument('--learning_rate_gen', type=float, default=1e-4, help='initial learning rate for training network.')
parser.add_argument('--learning_rate_disc', type=float, default=1e-5, help='initial learning rate for training network.')
parser.add_argument('--decay_factor', type=int, default=-0, help='decay factor for learning rate.')
parser.add_argument('--decay_step', type=int, default=30, help='if decay type is [stair].')
parser.add_argument('--decay_type', default='log', choices=['log', 'stair'], help='decay type : [log] and [stair].')

parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay for regularization')

parser.add_argument('--input_weight', type=float, default=1.0, help='input weight.')

parser.add_argument('--load_y_size', type=int, default=256, help='load size = the y-axis.')
parser.add_argument('--load_x_size', type=int, default=256, help='load size = the x-axis.')
parser.add_argument('--load_ch_size', type=int, default=3, help='load size = the z-axis.')

parser.add_argument('--input_y_size', type=int, default=256, help='input size = the y-axis.')
parser.add_argument('--input_x_size', type=int, default=256, help='input size = the x-axis.')
parser.add_argument('--input_ch_size', type=int, default=3, help='input size = the z-axis.')

parser.add_argument('--patch_y_size', type=int, default=256, help='patch size = the y-axis.')
parser.add_argument('--patch_x_size', type=int, default=256, help='patch size = the x-axis.')
parser.add_argument('--patch_ch_size', type=int, default=3, help='patch size = the z-axis.')

parser.add_argument('--output_y_size', type=int, default=256, help='input size = the y-axis.')
parser.add_argument('--output_x_size', type=int, default=256, help='input size = the x-axis.')
parser.add_argument('--output_ch_size', type=int, default=3, help='input size = the z-axis.')

parser.add_argument('--kernel_y_size', type=int, default=0, help='kernel size = the y-axis.')
parser.add_argument('--kernel_x_size', type=int, default=0, help='kernel size = the x-axis.')
parser.add_argument('--kernel_ch_size', type=int, default=0, help='kernel size = the z-axis.')

parser.add_argument('--gen_weight', type=float, default=1e0, help='weight on L2 term for generator gradient.')
parser.add_argument('--disc_weight', type=float, default=1e-1, help='weight on GAN term for generator gradient.')

parser.add_argument('--data_type', default='float32', help='data type : [float32] and [complex].')
parser.add_argument('--device', type=int, default=0, help='device used to train network')

parser.add_argument('--save_freq', type=int, default=10, help='save model every save_freq steps, 0 to disable')

PARSER = Parser(parser)
LOGGER = Logger(name=__name__).get_logger()

def main():
    PARAMS = PARSER.get_arguments()
    PARSER.write_args()
    LOGGER.info('GET the arguments')

    if PARAMS.device == None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(PARAMS.device)

    MODELS = Models(PARAMS)
    DATABASE = Database(PARAMS)
    TRAINERS = Trainer(PARAMS)
    LOGGER.info('CREATE the instances')

    LOGGER.info('START the ' + PARAMS.mode)
    if PARAMS.mode == 'train':
        TRAINERS.train(MODELS, DATABASE)
    elif PARAMS.mode == 'test':
        TRAINERS.test(MODELS, DATABASE)
    LOGGER.info('FINISH the ' + PARAMS.mode)

if __name__ == '__main__':
    main()
