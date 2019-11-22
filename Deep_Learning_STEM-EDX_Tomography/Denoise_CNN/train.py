import os

import tensorflow as tf
import numpy as np
import scipy.io as sio
from tqdm import tqdm

from utils import Logger
from utils import Parser

LOGGER = Logger(name=__name__).get_logger()
EPS = 1e-12

class Trainer:
    def __init__(self, param):
        self.__param = param

        self.scope = param.scope
        self.train_continue = param.train_continue
        self.checkpoint_dir = param.checkpoint_dir
        self.log_dir = param.log_dir

        self.checkpoint_id = param.checkpoint_id

        self.output_dir = param.output_dir

        self.network_type = param.network_type
        self.optimizer = param.optimizer
        self.loss = param.loss

        self.global_step = 0
        self.epoch_num = param.epoch_num
        self.batch_size = param.batch_size

        self.input_weight = param.input_weight

        self.input_y_size = param.input_y_size
        self.input_x_size = param.input_x_size
        self.input_ch_size = param.input_ch_size

        self.output_y_size = param.output_y_size
        self.output_x_size = param.output_x_size
        self.output_ch_size = param.output_ch_size

        if param.patch_y_size == -1:
            self.patch_y_size = param.input_y_size
        else:
            self.patch_y_size = param.patch_y_size

        if param.patch_x_size == -1:
            self.patch_x_size = param.input_x_size
        else:
            self.patch_x_size = param.patch_x_size

        if param.patch_ch_size == -1:
            self.patch_ch_size = param.input_ch_size
        else:
            self.patch_ch_size = param.patch_ch_size

        if param.data_type == 'float32':
            self.data_type = tf.float32
            self.label_type = tf.float32
        elif param.data_type == 'float16':
            self.data_type = tf.float16
            self.label_type = tf.float16

        if self.loss == 'softmax':
            self.label_type = tf.int32

        self.decay_type = param.decay_type
        self.weight_decay = param.weight_decay

        self.learning_rate_gen = param.learning_rate_gen
        self.learning_rate_disc = param.learning_rate_disc
        self.decay_factor = param.decay_factor
        self.decay_step = param.decay_step

        if self.decay_type == 'log':
            self.learning_rate_gen_set = self.learning_rate_gen * np.logspace(0, self.decay_factor, num=self.epoch_num)
            self.learning_rate_disc_set = self.learning_rate_disc * np.logspace(0, self.decay_factor, num=self.epoch_num)
        elif self.decay_type == 'stair':
            self.learning_rate_gen_set = self.learning_rate_gen * np.power(1e-1, [i // self.decay_step for i in range(self.epoch_num)])
            self.learning_rate_disc_set = self.learning_rate_disc * np.power(1e-1, [i // self.decay_step for i in range(self.epoch_num)])

        self.gen_weight = param.gen_weight
        self.disc_weight = param.disc_weight

        self.beta = param.beta
        self.save_freq = param.save_freq

    def _summary_image(self, image_dict={}):
        _summary_image_merge = [tf.summary.image(key, tf.image.convert_image_dtype(image_dict[key], dtype=self.data_type, saturate=False), max_outputs=self.batch_size, family=key) for key in image_dict]
        summary_image_op = tf.summary.merge(_summary_image_merge)

        return summary_image_op

    def _summary_predict(self, image_dict={}):
        _summary_image_merge = [tf.summary.image(key, tf.image.convert_image_dtype(image_dict[key], dtype=tf.uint8, saturate=False), max_outputs=self.batch_size, family=key) for key in image_dict]
        summary_image_op = tf.summary.merge(_summary_image_merge)

        return summary_image_op

    def _summary_scalar(self, scalar_dict={}):
        _summary_scalar_merge = [tf.summary.scalar(key, scalar_dict[key], family=key) for key in scalar_dict]
        summary_scalar_op = tf.summary.merge(_summary_scalar_merge)

        return summary_scalar_op

    def save(self, saver, sess, step=0):
        checkpoint_dir = os.path.join(self.checkpoint_dir, self.scope)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        saver.save(sess, "{}/model.ckpt".format(checkpoint_dir), global_step=step)

    def load(self, saver, sess):
        checkpoint_dir = os.path.join(self.checkpoint_dir, self.scope)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        if ckpt and ckpt.model_checkpoint_path:

            print('checkpoint : %s' % (ckpt.all_model_checkpoint_paths[self.checkpoint_id]))

            saver.restore(sess, ckpt.all_model_checkpoint_paths[self.checkpoint_id])
            self.global_step = int(ckpt.all_model_checkpoint_paths[self.checkpoint_id].split('/')[-1].split('-')[-1])
            return True
        else:
            return False

    def test(self, MODELS, DATABASE):
        '''
        if [training] is True, train_mode
        else [training] is False, validation_ and test_mode
        '''

        ''' 
        construct the MODEL, LOSS, and OPTIMIZER for training
        '''
        training_op = tf.placeholder(tf.bool, name='training')
        input_op = tf.placeholder(self.data_type, shape=[None, self.patch_y_size, self.patch_x_size, self.patch_ch_size], name='input')

        with tf.name_scope("generator_cgan"):
            output_op = MODELS.get_network(input=input_op, training=training_op, network_type=self.network_type, reuse=False, name='generator')

        LOGGER.info('CONSTRUCT the model for training')

        output_dir = self.output_dir
        if not os.path.exists(output_dir):
            LOGGER.info('CREATE the directory for saving the results')
            os.makedirs(output_dir)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            LOGGER.info('INITIALIZE the global variables')

            saver = tf.train.Saver()
            if self.load(saver, sess):
                LOGGER.info("[*] LOAD SUCCESS")
            else:
                LOGGER.info("[!] LOAD failed...")

            '''
            Test mode
            '''
            test_idx = DATABASE.get_index(shuffle=False, type='test')
            test_num = len(test_idx)

            ttest = tqdm(range(test_num))
            for itest in ttest:
                input = DATABASE.get_data(itest, mode='test', type='input')
                batch, batch_num = DATABASE.get_image2patch(input)

                for st in range(0, batch_num, self.batch_size):
                    ed = st + self.batch_size if st + self.batch_size < batch_num else batch_num

                    test_input = self.preprocess(self.input_weight * batch[st:ed, :, :, :])
                    test_output = sess.run(output_op, feed_dict={input_op: test_input, training_op: False})
                    batch[st:ed, :, :, 0:self.output_ch_size] = self.deprocess(test_output / self.input_weight)

                output, _ = DATABASE.get_patch2image(batch[:, :, :, 0:self.output_ch_size], type='cosin')
                sio.savemat(os.path.join(output_dir, 'output_%04d.mat' % itest), {'output': output})

    def train(self, MODELS, DATABASE):
        ''' 
        if [training] is True, train_mode
        else [training] is False, validation_ and test_mode
        '''
        training_op = tf.placeholder(tf.bool, name='training')
        learning_rate_gen_op = tf.placeholder(self.data_type, name='learning_rate_gen')
        learning_rate_disc_op = tf.placeholder(self.data_type, name='learning_rate_disc')

        gen_loss_mean_op = tf.placeholder(self.data_type, name='mean_gen_loss')
        disc_loss_mean_op = tf.placeholder(self.data_type, name='mean_disc_loss')

        LOGGER.info('CONSTRUCT the hyper parameters')

        ''' 
        construct the MODEL, LOSS, and OPTIMIZER for training
        '''
        label_op = tf.placeholder(self.label_type, shape=[None, self.patch_y_size, self.patch_x_size, self.output_ch_size], name='label')
        input_op = tf.placeholder(self.data_type, shape=[None, self.patch_y_size, self.patch_x_size, self.patch_ch_size], name='input')

        with tf.name_scope("generator_cgan"):
            output_op = MODELS.get_network(input=input_op, training=training_op, network_type=self.network_type, reuse=False, name='generator')

        with tf.name_scope("discriminator_real"):
            input_real_op = tf.concat([input_op, label_op], axis=3)
            pred_real_op = MODELS.get_network(input=input_real_op, training=training_op, network_type='discriminator', reuse=False, name='discriminator')
        with tf.name_scope("discriminator_fake"):
            input_fake_op = tf.concat([input_op, output_op], axis=3)
            pred_fake_op = MODELS.get_network(input=input_fake_op, training=training_op, network_type='discriminator', reuse=True, name='discriminator')

        with tf.name_scope("label_summary"):
            label_sum_op = self.deprocess(label_op)
        with tf.name_scope("input_summary"):
            input_sum_op = self.deprocess(input_op)
        with tf.name_scope("output_summary"):
            output_sum_op = self.deprocess(output_op / self.input_weight)

        LOGGER.info('CONSTRUCT the model for training')
        OPTIMIZERS = Optimizers(self.get_param())

        with tf.name_scope('gen_losses'):
            gen_l2_loss_op = 0.5 * tf.reduce_mean(tf.square(label_op - output_op))
            gen_gan_loss_op = tf.reduce_mean(-tf.log(pred_fake_op + EPS))
            reg_loss_op = tf.reduce_sum(tf.losses.get_regularization_losses())

            if self.weight_decay == 0.0:
                gen_loss_op = self.gen_weight * gen_l2_loss_op + self.disc_weight * gen_gan_loss_op
            else:
                gen_loss_op = self.gen_weight * gen_l2_loss_op + self.disc_weight * gen_gan_loss_op + self.weight_decay * reg_loss_op

        with tf.name_scope('disc_losses'):
            disc_loss_op = tf.reduce_mean(-(tf.log(pred_real_op + EPS) + tf.log(1 - pred_fake_op + EPS)))

        disc_vars = MODELS.get_vars(name='discriminator')
        disc_optim_op = OPTIMIZERS.get_optimizer(loss=disc_loss_op, var_list=disc_vars, optimizer_type=self.optimizer, beta=self.beta, learning_rate=learning_rate_disc_op, name='optimizer_disciminator')

        gen_vars = MODELS.get_vars(name='generator')
        gen_optim_op = OPTIMIZERS.get_optimizer(loss=gen_loss_op, var_list=gen_vars, optimizer_type=self.optimizer, beta=self.beta, learning_rate=learning_rate_gen_op, name='optimizer_generator')

        LOGGER.info('CONSTRUCT the loss & optimizers for training')

        ''' 
        create tensorboard summary
        '''
        with tf.name_scope('summaries'):
            summary_gen_op = self._summary_image(image_dict={'label_img': label_sum_op, 'input_img': input_sum_op, 'output_img': output_sum_op})
            summary_disc_op = self._summary_predict(image_dict={'disc_real': pred_real_op, 'disc_fake': pred_fake_op})
            summary_loss_op = self._summary_scalar(scalar_dict={'generator_loss': gen_loss_mean_op, 'discriminator_loss': disc_loss_mean_op})

        train_log_dir = os.path.join(self.log_dir, self.scope, 'train')
        val_log_dir = os.path.join(self.log_dir, self.scope, 'val')

        LOGGER.info('CONSTRUCT the tensorboard')

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            train_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
            val_summary_writer = tf.summary.FileWriter(val_log_dir)

            init = tf.global_variables_initializer()
            sess.run(init)

            LOGGER.info('INITIALIZE the global variables')

            saver = tf.train.Saver()
            if self.train_continue:
                if self.load(saver, sess):
                    LOGGER.info("[*] LOAD SUCCESS")
                else:
                    LOGGER.info("[!] LOAD failed...")

            try:
                LOGGER.info('START the training network')
                tepoch = tqdm(range(self.global_step, self.epoch_num))
                for epoch in tepoch:
                    lr_gen = self.learning_rate_gen_set[epoch]
                    lr_disc = self.learning_rate_disc_set[epoch]

                    '''
                    Train mode
                    '''
                    train_idx = DATABASE.get_index(type='train')
                    train_num = len(train_idx)

                    train_disc_loss_mean = 0
                    train_gen_loss_mean = 0
                    train_cnt = 0

                    for st in range(0, train_num, self.batch_size):
                        ed = st + self.batch_size if st + self.batch_size < train_num else train_num

                        idx = train_idx[st:ed]
                        train_label, train_input = DATABASE.get_database(idx, mode='train')
                        train_label = self.preprocess(self.input_weight * train_label)
                        train_input = self.preprocess(self.input_weight * train_input)

                        # Update Discriminator network
                        _, train_disc_loss = sess.run([disc_optim_op, disc_loss_op],
                                                     feed_dict={label_op: train_label,
                                                                input_op: train_input,
                                                                training_op: True,
                                                                learning_rate_disc_op: lr_disc})
                        # Update Generator network
                        _, train_gen_loss = sess.run([gen_optim_op, gen_l2_loss_op],
                                                     feed_dict={label_op: train_label,
                                                                input_op: train_input,
                                                                training_op: True,
                                                                learning_rate_gen_op: lr_gen})

                        train_disc_loss_mean += train_disc_loss
                        train_gen_loss_mean += train_gen_loss

                        train_cnt += 1

                    '''
                    Add tensorboard for training mode 
                    '''
                    idx = train_idx[0:self.batch_size]
                    train_label, train_input = DATABASE.get_database(idx, mode='train')
                    train_label = self.preprocess(self.input_weight * train_label)
                    train_input = self.preprocess(self.input_weight * train_input)

                    train_summary_disc, train_summary_gen, train_summary_loss_mean = sess.run([summary_disc_op, summary_gen_op, summary_loss_op],
                                                                                              feed_dict={label_op: train_label,
                                                                                              input_op: train_input,
                                                                                              training_op: False,
                                                                                              disc_loss_mean_op: train_disc_loss_mean / train_cnt,
                                                                                              gen_loss_mean_op: train_gen_loss_mean / train_cnt})

                    train_summary_writer.add_summary(train_summary_disc, epoch)
                    train_summary_writer.add_summary(train_summary_gen, epoch)
                    train_summary_writer.add_summary(train_summary_loss_mean, epoch)

                    '''
                    Validation mode
                    '''
                    val_idx = DATABASE.get_index(type='val')
                    val_num = len(val_idx)

                    val_disc_loss_mean = 0
                    val_gen_loss_mean = 0
                    val_cnt = 0

                    for st in range(0, val_num, self.batch_size):
                        ed = st + self.batch_size if st + self.batch_size < val_num else val_num

                        idx = val_idx[st:ed]
                        val_label, val_input = DATABASE.get_database(idx, mode='val')
                        val_label = self.preprocess(self.input_weight * val_label)
                        val_input = self.preprocess(self.input_weight * val_input)

                        val_disc_loss, val_gen_loss = sess.run([disc_loss_op, gen_l2_loss_op],
                                                               feed_dict={label_op: val_label,
                                                                          input_op: val_input,
                                                                          training_op: False})

                        val_disc_loss_mean += val_disc_loss
                        val_gen_loss_mean += val_gen_loss
                        val_cnt += 1

                    '''
                    Add tensorboard for val mode 
                    '''
                    idx = val_idx[0:self.batch_size]
                    val_label, val_input = DATABASE.get_database(idx, mode='val')
                    val_label = self.preprocess(self.input_weight * val_label)
                    val_input = self.preprocess(self.input_weight * val_input)

                    val_summary_disc, val_summary_gen, val_summary_loss_mean = sess.run([summary_disc_op, summary_gen_op, summary_loss_op],
                                                                                        feed_dict={label_op: val_label,
                                                                                        input_op: val_input,
                                                                                        training_op: False,
                                                                                        disc_loss_mean_op: val_disc_loss_mean / val_cnt,
                                                                                        gen_loss_mean_op: val_gen_loss_mean / val_cnt})

                    val_summary_writer.add_summary(val_summary_disc, epoch)
                    val_summary_writer.add_summary(val_summary_gen, epoch)
                    val_summary_writer.add_summary(val_summary_loss_mean, epoch)

                    '''
                    Print losses
                    '''
                    tepoch.pos = 1
                    tepoch.set_postfix(train_gen='{:.6f}'.format(train_gen_loss_mean / train_cnt), train_disc='{:.6f}'.format(train_disc_loss_mean / train_cnt),
                                       val_gen='{:.6f}'.format(val_gen_loss_mean / val_cnt), val_disc='{:.6f}'.format(val_disc_loss_mean / val_cnt))
                    tepoch.pos = 0

                    if (epoch % self.save_freq) == 0:
                        self.save(saver, sess, epoch)

            finally:
                self.save(saver, sess, epoch+1)

    def get_param(self):
        return self.__param

    def preprocess(self, input):
        with tf.name_scope("preprocess"):
            return input * 2 - 1

    def deprocess(self, input):
        with tf.name_scope("deprocess"):
            return (input + 1) / 2

class Optimizers:
    def __init__(self, param):
        self.optimizer = param.optimizer

    def get_optimizer(self, loss, var_list, optimizer_type='adam', beta=0.9, learning_rate=1e-1, name='optimizer'):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.variable_scope(name):
            with tf.control_dependencies(update_ops):
                if optimizer_type == 'sgd':
                    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, var_list=var_list)
                elif optimizer_type == 'adam':
                    optimizer = tf.train.AdamOptimizer(learning_rate, beta).minimize(loss, var_list=var_list)
                elif optimizer_type == 'rmsprop':
                    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, var_list=var_list)

        return optimizer

class Losses:
    def __init__(self, param):
        self.loss = param.loss
        self.weight_decay = param.weight_decay

    def get_regularizations(self, name='variable_regularizations'):
        with tf.variable_scope(name):
            reg = tf.reduce_sum(tf.losses.get_regularization_losses())
        return reg

    def get_regularization(self, x, reg_type='l2'):
        with tf.variable_scope(reg_type + '_regularization'):
            if reg_type == 'l1':
                reg = tf.reduce_mean(tf.abs(x))
            elif reg_type == 'l2':
                reg = tf.reduce_mean(tf.square(x))
            elif reg_type == 'lp':
                reg = tf.norm(tensor=x, ord=ord)
            elif reg_type == 'tv':
                reg = tf.reduce_mean(tf.image.total_variation(x))
        return reg

    def get_l1_regularization(self, x, reg_type='l1'):
        with tf.variable_scope(reg_type + '_regularization'):
            reg = tf.reduce_mean(tf.abs(x))
        return reg

    def get_l2_regularization(self, x, reg_type='l2'):
        with tf.variable_scope(reg_type + '_regularization'):
            reg = tf.reduce_mean(tf.square(x))
        return reg

    def get_norm_regularization(self, x, ord=2, reg_type='lp'):
        with tf.variable_scope(reg_type + '_regularization'):
            reg = tf.norm(tensor=x, ord=ord)
        return reg

    def get_tv_regularization(self, x, reg_type='tv'):
        with tf.variable_scope(reg_type + '_regularization'):
            reg = tf.reduce_mean(tf.image.total_variation(x))
        return reg

    def get_loss(self, label, pred, loss_type='l2'):
        with tf.variable_scope(loss_type + '_loss'):
            if loss_type == 'l2':
                loss = tf.reduce_mean(tf.square(label - pred))
            elif loss_type == 'l1':
                loss = tf.reduce_mean(tf.abs(label - pred))
            elif loss_type == 'softmax':
                loss = tf.losses.sparse_softmax_cross_entropy(label, pred)
            elif loss_type == 'discriminator':
                loss = tf.reduce_mean(-(tf.log(label + EPS) + tf.log(1 - pred + EPS)))
        return loss