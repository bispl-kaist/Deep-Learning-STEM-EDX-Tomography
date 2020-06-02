import tensorflow as tf
EPS = 1e-12

class Models:

    def __init__(self, param):
        self.network_type = param.network_type
        self.learning_type = param.learning_type
        self.weight_decay = param.weight_decay

        if param.data_type == 'float32':
            self.data_type = tf.float32
        elif param.data_type == 'float16':
            self.data_type = tf.float16

        if param.regularizer == 'l1':
            self.regularizer = self.get_l1_regularizers
        elif param.regularizer == 'l2':
            self.regularizer = self.get_l2_regularizers
        elif param.regularizer == 'none':
            self.regularizer = 'none'

        if param.initializer == 'xavier':
            self.initializer = tf.keras.initializers.glorot_normal()
        elif param.initializer == 'he':
            self.initializer = tf.keras.initializers.he_normal()
        elif param.initializer == 'gaussian':
            self.initializer = tf.initializers.truncated_normal(stddev=0.02, dtype=self.data_type)
        elif param.initializer == 'normal':
            self.initializer = tf.random_normal_initializer(0, 0.02)

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


    '''
    GET REGULAIZERS
    '''
    def get_l1_regularizers(self, x):
        regularizer = tf.reduce_mean(tf.abs(x))
        return regularizer

    def get_l2_regularizers(self, x):
        regularizer = tf.reduce_mean(tf.square(x))
        return regularizer

    '''
    SELECT NETWORK STRUCTURE such as RES-NET and U-NET
    '''
    def get_network(self, input, training, network_type='unet', reuse=tf.AUTO_REUSE, name='unet'):
        with tf.variable_scope(name, reuse=reuse):
            if network_type == 'unet':
                output = self.get_unet(input=input, training=training)
            elif network_type == 'autoencoder':
                output = self.get_autoencoder(input=input, training=training)
            elif network_type == 'resnet':
                output = self.get_resnet(input=input, training=training)
            elif network_type == 'cgan':
                output = self.get_generator_cgan(input=input, training=training)
            elif network_type == 'cgan_w_avgp':
                output = self.get_generator_cgan_w_avgp(input=input, training=training)
            elif network_type == 'cgan_w_cnvt_avgp':
                output = self.get_generator_cgan_w_cnvt_avgp(input=input, training=training)
            elif network_type == 'cgan_w_avgp_cnvt':
                output = self.get_generator_cgan_w_avgp_cnvt(input=input, training=training)
            elif network_type == 'discriminator':
                output = self.get_discriminator(input=input, training=training)

        return output

    def get_vars(self, name='generator'):
        t_vars = tf.trainable_variables()
        vars = [var for var in t_vars if name in var.name]

        return vars

    '''
    RES-NET STRUCTURE
    '''
    def get_resnet(self, input, training):
        print('currently, it dose not implemented.')
        return []

    '''
    AUTOENCODER STRUCTURE
    '''
    def get_autoencoder(self, input, training):

        '''
        ENCODER PART
        '''
        stg = 0
        with tf.variable_scope('enc_{}'.format(stg)):
            enc0 = self.get_standard_block(input, training=training, filters=[64, 64], is_bnorm=True, is_activate=True)
            pool0 = self.get_pool2d(enc0)

        stg = 1
        with tf.variable_scope('enc_{}'.format(stg)):
            enc1 = self.get_standard_block(pool0, training=training, filters=[128, 128], is_bnorm=True, is_activate=True)
            pool1 = self.get_pool2d(enc1)

        stg = 2
        with tf.variable_scope('enc_{}'.format(stg)):
            enc2 = self.get_standard_block(pool1, training=training, filters=[256, 256], is_bnorm=True, is_activate=True)
            pool2 = self.get_pool2d(enc2)

        stg = 3
        with tf.variable_scope('enc_{}'.format(stg)):
            enc3 = self.get_standard_block(pool2, training=training, filters=[512, 512], is_bnorm=True, is_activate=True)
            pool3 = self.get_pool2d(enc3)

        '''
        ENCODER & DECODER PART
        '''
        stg = 4
        with tf.variable_scope('enc_dec_{}'.format(stg)):
            enc4 = self.get_standard_block(pool3, training=training, filters=[1024, 512], is_bnorm=True, is_activate=True)

        '''
        DECODER PART
        '''
        stg = 3
        with tf.variable_scope('dec_{}'.format(stg)):
            unpool3 = self.get_unpool2d(enc4)
            dec3 = self.get_standard_block(unpool3, training=training, filters=[512, 256], is_bnorm=True, is_activate=True)

        stg = 2
        with tf.variable_scope('dec_{}'.format(stg)):
            unpool2 = self.get_unpool2d(dec3)
            dec2 = self.get_standard_block(unpool2, training=training, filters=[256, 128], is_bnorm=True, is_activate=True)

        stg = 1
        with tf.variable_scope('dec_{}'.format(stg)):
            unpool1 = self.get_unpool2d(dec2)
            dec1 = self.get_standard_block(unpool1, training=training, filters=[128, 64], is_bnorm=True, is_activate=True)

        stg = 0
        with tf.variable_scope('dec_{}'.format(stg)):
            unpool0 = self.get_unpool2d(dec1)
            dec0 = self.get_standard_block(unpool0, training=training, filters=[64, 64], is_bnorm=True, is_activate=True)

        '''
        FULLY-CONNECTION PART
        '''
        with tf.variable_scope('fc'):
            output = self.get_standard_block(dec0, filters=[self.output_ch_size], kernel_size=[1, 1], is_bnorm=False, is_activate=False)

        if self.learning_type == 'residual':
            output = tf.add(output, input)

        return output

    '''
    U-NET STRUCTURE
    '''
    def get_unet(self, input, training):
        '''
        ENCODER PART
        '''
        stg = 0
        with tf.variable_scope('enc_{}'.format(stg)):
            enc0 = self.get_standard_block(input, kernel_size=[3, 3], stride=1, leak=0.2, filters=[64, 64], training=training, is_bnorm=True, is_activate=True)
            pool0 = self.get_pool2d(enc0)

        stg = 1
        with tf.variable_scope('enc_{}'.format(stg)):
            enc1 = self.get_standard_block(pool0, kernel_size=[3, 3], stride=1, leak=0.2, filters=[128, 128], training=training, is_bnorm=True, is_activate=True)
            pool1 = self.get_pool2d(enc1)

        stg = 2
        with tf.variable_scope('enc_{}'.format(stg)):
            enc2 = self.get_standard_block(pool1, kernel_size=[3, 3], stride=1, leak=0.2, filters=[256, 256], training=training, is_bnorm=True, is_activate=True)
            pool2 = self.get_pool2d(enc2)

        stg = 3
        with tf.variable_scope('enc_{}'.format(stg)):
            enc3 = self.get_standard_block(pool2, kernel_size=[3, 3], stride=1, leak=0.2, filters=[512, 512], training=training, is_bnorm=True, is_activate=True)
            pool3 = self.get_pool2d(enc3)

        '''
        ENCODER & DECODER PART
        '''
        stg = 4
        with tf.variable_scope('enc_{}'.format(stg)):
            enc4 = self.get_standard_block(pool3, kernel_size=[3, 3], stride=1, leak=0.2, filters=[1024], training=training, is_bnorm=True, is_activate=True)

        with tf.variable_scope('dec_{}'.format(stg)):
            dec4 = self.get_standard_block(enc4, kernel_size=[3, 3], stride=1, leak=0.0, filters=[512], training=training, is_bnorm=True, is_activate=True)

        '''
        DECODER PART
        '''
        stg = 3
        with tf.variable_scope('dec_{}'.format(stg)):
            unpool3 = self.get_unpool2d(dec4)
            concat3 = self.get_concat([unpool3, enc3])
            dec3 = self.get_standard_block(concat3, kernel_size=[3, 3], stride=1, leak=0.0, filters=[512, 256], training=training, is_bnorm=True, is_activate=True)

        stg = 2
        with tf.variable_scope('dec_{}'.format(stg)):
            unpool2 = self.get_unpool2d(dec3)
            concat2 = self.get_concat([unpool2, enc2])
            dec2 = self.get_standard_block(concat2, kernel_size=[3, 3], stride=1, leak=0.0, filters=[256, 128], training=training, is_bnorm=True, is_activate=True)

        stg = 1
        with tf.variable_scope('dec_{}'.format(stg)):
            unpool1 = self.get_unpool2d(dec2)
            concat1 = self.get_concat([unpool1, enc1])
            dec1 = self.get_standard_block(concat1, kernel_size=[3, 3], stride=1, leak=0.0, filters=[128, 64], training=training, is_bnorm=True, is_activate=True)

        stg = 0
        with tf.variable_scope('dec_{}'.format(stg)):
            unpool0 = self.get_unpool2d(dec1)
            concat0 = self.get_concat([unpool0, enc0])
            dec0 = self.get_standard_block(concat0, kernel_size=[3, 3], stride=1, leak=0.0, filters=[64, 64], training=training, is_bnorm=True, is_activate=True)

        '''
        FULLY-CONNECTION PART
        '''
        with tf.variable_scope('fc'):
            output = self.get_standard_block(dec0, kernel_size=[1, 1], stride=1, leak=0.0, filters=[self.output_ch_size], training=training, is_bnorm=False, is_activate=False)

        if self.learning_type == 'residual':
            with tf.variable_scope('add'):
                output = tf.add(output, input)

        return output

    '''
       CGAN : U-NET STRUCTURE
       '''

    def get_generator_cgan(self, input, training):
        '''
        ENCODER PART
        '''
        stg = 0
        with tf.variable_scope('enc_{}'.format(stg)):
            enc0 = self.get_standard_block(input, kernel_size=[4, 4], stride=[2, 2], leak=0.2, filters=[64],
                                           training=training, is_bnorm=False, is_activate=True)

        stg = 1
        with tf.variable_scope('enc_{}'.format(stg)):
            enc1 = self.get_standard_block(enc0, kernel_size=[4, 4], stride=[2, 2], leak=0.2, filters=[64 * 2],
                                           training=training, is_bnorm=True, is_activate=True)

        stg = 2
        with tf.variable_scope('enc_{}'.format(stg)):
            enc2 = self.get_standard_block(enc1, kernel_size=[4, 4], stride=[2, 2], leak=0.2, filters=[64 * 4],
                                           training=training, is_bnorm=True, is_activate=True)

        stg = 3
        with tf.variable_scope('enc_{}'.format(stg)):
            enc3 = self.get_standard_block(enc2, kernel_size=[4, 4], stride=[2, 2], leak=0.2, filters=[64 * 8],
                                           training=training, is_bnorm=True, is_activate=True)

        stg = 4
        with tf.variable_scope('enc_{}'.format(stg)):
            enc4 = self.get_standard_block(enc3, kernel_size=[4, 4], stride=[2, 2], leak=0.2, filters=[64 * 8],
                                           training=training, is_bnorm=True, is_activate=True)

        stg = 5
        with tf.variable_scope('enc_{}'.format(stg)):
            enc5 = self.get_standard_block(enc4, kernel_size=[4, 4], stride=[2, 2], leak=0.2, filters=[64 * 8],
                                           training=training, is_bnorm=True, is_activate=True)

        stg = 6
        with tf.variable_scope('enc_{}'.format(stg)):
            enc6 = self.get_standard_block(enc5, kernel_size=[4, 4], stride=[2, 2], leak=0.2, filters=[64 * 8],
                                           training=training, is_bnorm=True, is_activate=False)

        '''
        DECODER PART
        '''

        stg = 6
        with tf.variable_scope('dec_{}'.format(stg)):
            concat6 = enc6
            dec5 = self.get_standard_deblock(concat6, kernel_size=[4, 4], stride=[2, 2], leak=0.0, filters=[64 * 8],
                                             training=training, is_bnorm=True, is_activate=True)

            dec5 = tf.nn.dropout(dec5, keep_prob=1 - tf.cast(training, tf.float32) * 0.5)
            concat5 = self.get_concat([dec5, enc5], axis=3)

        stg = 5
        with tf.variable_scope('dec_{}'.format(stg)):
            dec4 = self.get_standard_deblock(concat5, kernel_size=[4, 4], stride=[2, 2], leak=0.0, filters=[64 * 8],
                                             training=training, is_bnorm=True, is_activate=True)

            dec4 = tf.nn.dropout(dec4, keep_prob=1 - tf.cast(training, tf.float32) * 0.5)
            concat4 = self.get_concat([dec4, enc4], axis=3)

        stg = 4
        with tf.variable_scope('dec_{}'.format(stg)):
            dec3 = self.get_standard_deblock(concat4, kernel_size=[4, 4], stride=[2, 2], leak=0.0, filters=[64 * 8],
                                             training=training, is_bnorm=True, is_activate=True)

            concat3 = self.get_concat([dec3, enc3], axis=3)

        stg = 3
        with tf.variable_scope('dec_{}'.format(stg)):
            dec2 = self.get_standard_deblock(concat3, kernel_size=[4, 4], stride=[2, 2], leak=0.0, filters=[64 * 4],
                                             training=training, is_bnorm=True, is_activate=True)

            concat2 = self.get_concat([dec2, enc2], axis=3)

        stg = 2
        with tf.variable_scope('dec_{}'.format(stg)):
            dec1 = self.get_standard_deblock(concat2, kernel_size=[4, 4], stride=[2, 2], leak=0.0, filters=[64 * 2],
                                             training=training, is_bnorm=True, is_activate=True)

            concat1 = self.get_concat([dec1, enc1], axis=3)

        stg = 1
        with tf.variable_scope('dec_{}'.format(stg)):
            dec0 = self.get_standard_deblock(concat1, kernel_size=[4, 4], stride=[2, 2], leak=0.0, filters=[64],
                                             training=training, is_bnorm=True, is_activate=True)

            concat0 = self.get_concat([dec0, enc0], axis=3)

        stg = 0
        with tf.variable_scope('dec_{}'.format(stg)):
            output = self.get_standard_deblock(concat0, kernel_size=[4, 4], stride=[2, 2], leak=0.0, filters=[self.output_ch_size],
                                               training=training, is_bnorm=False, is_activate=True)
        '''
        FULLY-CONNECTION PART
        '''
        with tf.variable_scope('tanh'):
            output = tf.tanh(output)

        if self.learning_type == 'residual':
            with tf.variable_scope('add'):
                output = tf.add(output, input)

        return output


    def get_generator_cgan_w_avgp(self, input, training):
        '''
        ENCODER PART
        '''
        stg = 0
        with tf.variable_scope('enc_{}'.format(stg)):
            enc0 = self.get_standard_block(input, kernel_size=[4, 4], stride=[2, 2], leak=0.2, filters=[64],
                                           training=training, is_bnorm=False, is_activate=True)

        stg = 1
        with tf.variable_scope('enc_{}'.format(stg)):
            enc1 = self.get_standard_block(enc0, kernel_size=[4, 4], stride=[1, 1], leak=0.2, filters=[64 * 2],
                                           training=training, is_bnorm=True, is_activate=True)
            enc1 = self.get_pool2d(enc1)

        stg = 2
        with tf.variable_scope('enc_{}'.format(stg)):
            enc2 = self.get_standard_block(enc1, kernel_size=[4, 4], stride=[1, 1], leak=0.2, filters=[64 * 4],
                                           training=training, is_bnorm=True, is_activate=True)
            enc2 = self.get_pool2d(enc2)

        stg = 3
        with tf.variable_scope('enc_{}'.format(stg)):
            enc3 = self.get_standard_block(enc2, kernel_size=[4, 4], stride=[1, 1], leak=0.2, filters=[64 * 8],
                                           training=training, is_bnorm=True, is_activate=True)
            enc3 = self.get_pool2d(enc3)

        stg = 4
        with tf.variable_scope('enc_{}'.format(stg)):
            enc4 = self.get_standard_block(enc3, kernel_size=[4, 4], stride=[1, 1], leak=0.2, filters=[64 * 8],
                                           training=training, is_bnorm=True, is_activate=True)
            enc4 = self.get_pool2d(enc4)

        stg = 5
        with tf.variable_scope('enc_{}'.format(stg)):
            enc5 = self.get_standard_block(enc4, kernel_size=[4, 4], stride=[1, 1], leak=0.2, filters=[64 * 8],
                                           training=training, is_bnorm=True, is_activate=True)
            enc5 = self.get_pool2d(enc5)

        stg = 6
        with tf.variable_scope('enc_{}'.format(stg)):
            enc6 = self.get_standard_block(enc5, kernel_size=[4, 4], stride=[1, 1], leak=0.2, filters=[64 * 8],
                                           training=training, is_bnorm=True, is_activate=False)
            enc6 = self.get_pool2d(enc6)

        '''
        DECODER PART
        '''

        stg = 6
        with tf.variable_scope('dec_{}'.format(stg)):
            concat6 = enc6
            dec5 = self.get_standard_deblock(concat6, kernel_size=[4, 4], stride=[1, 1], leak=0.0, filters=[64 * 8],
                                             training=training, is_bnorm=True, is_activate=True)

            dec5 = tf.nn.dropout(dec5, keep_prob=1 - tf.cast(training, tf.float32) * 0.5)
            dec5 = self.get_unpool2d(dec5)

        stg = 5
        with tf.variable_scope('dec_{}'.format(stg)):
            concat5 = self.get_concat([dec5, enc5], axis=3)
            dec4 = self.get_standard_deblock(concat5, kernel_size=[4, 4], stride=[1, 1], leak=0.0, filters=[64 * 8],
                                             training=training, is_bnorm=True, is_activate=True)

            dec4 = tf.nn.dropout(dec4, keep_prob=1 - tf.cast(training, tf.float32) * 0.5)
            dec4 = self.get_unpool2d(dec4)

        stg = 4
        with tf.variable_scope('dec_{}'.format(stg)):
            concat4 = self.get_concat([dec4, enc4], axis=3)
            dec3 = self.get_standard_deblock(concat4, kernel_size=[4, 4], stride=[1, 1], leak=0.0, filters=[64 * 8],
                                             training=training, is_bnorm=True, is_activate=True)

            dec3 = tf.nn.dropout(dec3, keep_prob=1 - tf.cast(training, tf.float32) * 0.5)
            dec3 = self.get_unpool2d(dec3)

        stg = 3
        with tf.variable_scope('dec_{}'.format(stg)):
            concat3 = self.get_concat([dec3, enc3], axis=3)
            dec2 = self.get_standard_deblock(concat3, kernel_size=[4, 4], stride=[1, 1], leak=0.0, filters=[64 * 4],
                                             training=training, is_bnorm=True, is_activate=True)
            dec2 = self.get_unpool2d(dec2)

        stg = 2
        with tf.variable_scope('dec_{}'.format(stg)):
            concat2 = self.get_concat([dec2, enc2], axis=3)
            dec1 = self.get_standard_deblock(concat2, kernel_size=[4, 4], stride=[1, 1], leak=0.0, filters=[64 * 2],
                                             training=training, is_bnorm=True, is_activate=True)

        stg = 1
        with tf.variable_scope('dec_{}'.format(stg)):
            dec1 = self.get_unpool2d(dec1)
            concat1 = self.get_concat([dec1, enc1], axis=3)
            dec0 = self.get_standard_deblock(concat1, kernel_size=[4, 4], stride=[1, 1], leak=0.0, filters=[64],
                                             training=training, is_bnorm=True, is_activate=True)
            dec0 = self.get_unpool2d(dec0)

        stg = 0
        with tf.variable_scope('dec_{}'.format(stg)):
            concat0 = self.get_concat([dec0, enc0], axis=3)
            output = self.get_standard_deblock(concat0, kernel_size=[4, 4], stride=[2, 2], leak=0.0, filters=[self.output_ch_size],
                                               training=training, is_bnorm=False, is_activate=True)

        '''
        FULLY-CONNECTION PART
        '''
        with tf.variable_scope('tanh'):
            output = tf.tanh(output)

        if self.learning_type == 'residual':
            with tf.variable_scope('add'):
                output = tf.add(output, input)

        return output

    '''
       CGAN : U-NET STRUCTURE
       '''

    def get_generator_cgan_w_cnvt_avgp(self, input, training):
        '''
        ENCODER PART
        '''
        stg = 0
        with tf.variable_scope('enc_{}'.format(stg)):
            enc0 = self.get_standard_block(input, kernel_size=[4, 4], stride=[2, 2], leak=0.2, filters=[64],
                                           training=training, is_bnorm=False, is_activate=True)

        stg = 1
        with tf.variable_scope('enc_{}'.format(stg)):
            enc1 = self.get_standard_block(enc0, kernel_size=[4, 4], stride=[2, 2], leak=0.2, filters=[64 * 2],
                                           training=training, is_bnorm=True, is_activate=True)

        stg = 2
        with tf.variable_scope('enc_{}'.format(stg)):
            enc2 = self.get_standard_block(enc1, kernel_size=[4, 4], stride=[2, 2], leak=0.2, filters=[64 * 4],
                                           training=training, is_bnorm=True, is_activate=True)

        stg = 3
        with tf.variable_scope('enc_{}'.format(stg)):
            enc3 = self.get_standard_block(enc2, kernel_size=[4, 4], stride=[2, 2], leak=0.2, filters=[64 * 8],
                                           training=training, is_bnorm=True, is_activate=True)

        stg = 4
        with tf.variable_scope('enc_{}'.format(stg)):
            enc4 = self.get_standard_block(enc3, kernel_size=[4, 4], stride=[2, 2], leak=0.2, filters=[64 * 8],
                                           training=training, is_bnorm=True, is_activate=True)

        stg = 5
        with tf.variable_scope('enc_{}'.format(stg)):
            enc5 = self.get_standard_block(enc4, kernel_size=[4, 4], stride=[2, 2], leak=0.2, filters=[64 * 8],
                                           training=training, is_bnorm=True, is_activate=True)
        stg = 6
        with tf.variable_scope('enc_{}'.format(stg)):
            enc6 = self.get_standard_block(enc5, kernel_size=[4, 4], stride=[2, 2], leak=0.2, filters=[64 * 8],
                                           training=training, is_bnorm=True, is_activate=False)

        '''
        DECODER PART
        '''

        stg = 6
        with tf.variable_scope('dec_{}'.format(stg)):
            concat6 = enc6
            dec5 = self.get_standard_deblock(concat6, kernel_size=[4, 4], stride=[1, 1], leak=0.0, filters=[64 * 8],
                                             training=training, is_bnorm=True, is_activate=True)

            dec5 = tf.nn.dropout(dec5, keep_prob=1 - tf.cast(training, tf.float32) * 0.5)
            dec5 = self.get_unpool2d(dec5)

        stg = 5
        with tf.variable_scope('dec_{}'.format(stg)):
            concat5 = self.get_concat([dec5, enc5], axis=3)
            dec4 = self.get_standard_deblock(concat5, kernel_size=[4, 4], stride=[1, 1], leak=0.0, filters=[64 * 8],
                                             training=training, is_bnorm=True, is_activate=True)

            dec4 = tf.nn.dropout(dec4, keep_prob=1 - tf.cast(training, tf.float32) * 0.5)
            dec4 = self.get_unpool2d(dec4)

        stg = 4
        with tf.variable_scope('dec_{}'.format(stg)):
            concat4 = self.get_concat([dec4, enc4], axis=3)
            dec3 = self.get_standard_deblock(concat4, kernel_size=[4, 4], stride=[1, 1], leak=0.0, filters=[64 * 8],
                                             training=training, is_bnorm=True, is_activate=True)

            dec3 = tf.nn.dropout(dec3, keep_prob=1 - tf.cast(training, tf.float32) * 0.5)
            dec3 = self.get_unpool2d(dec3)

        stg = 3
        with tf.variable_scope('dec_{}'.format(stg)):
            concat3 = self.get_concat([dec3, enc3], axis=3)
            dec2 = self.get_standard_deblock(concat3, kernel_size=[4, 4], stride=[1, 1], leak=0.0, filters=[64 * 4],
                                             training=training, is_bnorm=True, is_activate=True)
            dec2 = self.get_unpool2d(dec2)

        stg = 2
        with tf.variable_scope('dec_{}'.format(stg)):
            concat2 = self.get_concat([dec2, enc2], axis=3)
            dec1 = self.get_standard_deblock(concat2, kernel_size=[4, 4], stride=[1, 1], leak=0.0, filters=[64 * 2],
                                             training=training, is_bnorm=True, is_activate=True)

        stg = 1
        with tf.variable_scope('dec_{}'.format(stg)):
            dec1 = self.get_unpool2d(dec1)
            concat1 = self.get_concat([dec1, enc1], axis=3)
            dec0 = self.get_standard_deblock(concat1, kernel_size=[4, 4], stride=[1, 1], leak=0.0, filters=[64],
                                             training=training, is_bnorm=True, is_activate=True)
            dec0 = self.get_unpool2d(dec0)

        stg = 0
        with tf.variable_scope('dec_{}'.format(stg)):
            concat0 = self.get_concat([dec0, enc0], axis=3)
            output = self.get_standard_deblock(concat0, kernel_size=[4, 4], stride=[2, 2], leak=0.0, filters=[self.output_ch_size],
                                               training=training, is_bnorm=False, is_activate=True)
        '''
        FULLY-CONNECTION PART
        '''
        with tf.variable_scope('tanh'):
            output = tf.tanh(output)

        if self.learning_type == 'residual':
            with tf.variable_scope('add'):
                output = tf.add(output, input)

        return output

    '''
       CGAN : U-NET STRUCTURE
       '''

    def get_generator_cgan_w_avgp_cnvt(self, input, training):
        '''
        ENCODER PART
        '''
        stg = 0
        with tf.variable_scope('enc_{}'.format(stg)):
            enc0 = self.get_standard_block(input, kernel_size=[4, 4], stride=[2, 2], leak=0.2, filters=[64],
                                           training=training, is_bnorm=False, is_activate=True)

        stg = 1
        with tf.variable_scope('enc_{}'.format(stg)):
            enc1 = self.get_standard_block(enc0, kernel_size=[4, 4], stride=[1, 1], leak=0.2, filters=[64 * 2],
                                           training=training, is_bnorm=True, is_activate=True)
            enc1 = self.get_pool2d(enc1)

        stg = 2
        with tf.variable_scope('enc_{}'.format(stg)):
            enc2 = self.get_standard_block(enc1, kernel_size=[4, 4], stride=[1, 1], leak=0.2, filters=[64 * 4],
                                           training=training, is_bnorm=True, is_activate=True)
            enc2 = self.get_pool2d(enc2)

        stg = 3
        with tf.variable_scope('enc_{}'.format(stg)):
            enc3 = self.get_standard_block(enc2, kernel_size=[4, 4], stride=[1, 1], leak=0.2, filters=[64 * 8],
                                           training=training, is_bnorm=True, is_activate=True)
            enc3 = self.get_pool2d(enc3)

        stg = 4
        with tf.variable_scope('enc_{}'.format(stg)):
            enc4 = self.get_standard_block(enc3, kernel_size=[4, 4], stride=[1, 1], leak=0.2, filters=[64 * 8],
                                           training=training, is_bnorm=True, is_activate=True)
            enc4 = self.get_pool2d(enc4)

        stg = 5
        with tf.variable_scope('enc_{}'.format(stg)):
            enc5 = self.get_standard_block(enc4, kernel_size=[4, 4], stride=[1, 1], leak=0.2, filters=[64 * 8],
                                           training=training, is_bnorm=True, is_activate=True)
            enc5 = self.get_pool2d(enc5)

        stg = 6
        with tf.variable_scope('enc_{}'.format(stg)):
            enc6 = self.get_standard_block(enc5, kernel_size=[4, 4], stride=[1, 1], leak=0.2, filters=[64 * 8],
                                           training=training, is_bnorm=True, is_activate=False)
            enc6 = self.get_pool2d(enc6)

        '''
        DECODER PART
        '''

        stg = 6
        with tf.variable_scope('dec_{}'.format(stg)):
            concat6 = enc6
            dec5 = self.get_standard_deblock(concat6, kernel_size=[4, 4], stride=[2, 2], leak=0.0, filters=[64 * 8],
                                             training=training, is_bnorm=True, is_activate=True)

            dec5 = tf.nn.dropout(dec5, keep_prob=1 - tf.cast(training, tf.float32) * 0.5)

        stg = 5
        with tf.variable_scope('dec_{}'.format(stg)):

            concat5 = self.get_concat([dec5, enc5], axis=3)
            dec4 = self.get_standard_deblock(concat5, kernel_size=[4, 4], stride=[2, 2], leak=0.0, filters=[64 * 8],
                                             training=training, is_bnorm=True, is_activate=True)

            dec4 = tf.nn.dropout(dec4, keep_prob=1 - tf.cast(training, tf.float32) * 0.5)

        stg = 4
        with tf.variable_scope('dec_{}'.format(stg)):
            concat4 = self.get_concat([dec4, enc4], axis=3)
            dec3 = self.get_standard_deblock(concat4, kernel_size=[4, 4], stride=[2, 2], leak=0.0, filters=[64 * 8],
                                             training=training, is_bnorm=True, is_activate=True)

            dec3 = tf.nn.dropout(dec3, keep_prob=1 - tf.cast(training, tf.float32) * 0.5)

        stg = 3
        with tf.variable_scope('dec_{}'.format(stg)):
            concat3 = self.get_concat([dec3, enc3], axis=3)
            dec2 = self.get_standard_deblock(concat3, kernel_size=[4, 4], stride=[2, 2], leak=0.0, filters=[64 * 4],
                                             training=training, is_bnorm=True, is_activate=True)

        stg = 2
        with tf.variable_scope('dec_{}'.format(stg)):
            concat2 = self.get_concat([dec2, enc2], axis=3)
            dec1 = self.get_standard_deblock(concat2, kernel_size=[4, 4], stride=[2, 2], leak=0.0, filters=[64 * 2],
                                             training=training, is_bnorm=True, is_activate=True)

        stg = 1
        with tf.variable_scope('dec_{}'.format(stg)):
            concat1 = self.get_concat([dec1, enc1], axis=3)
            dec0 = self.get_standard_deblock(concat1, kernel_size=[4, 4], stride=[2, 2], leak=0.0, filters=[64],
                                             training=training, is_bnorm=True, is_activate=True)

        stg = 0
        with tf.variable_scope('dec_{}'.format(stg)):
            concat0 = self.get_concat([dec0, enc0], axis=3)
            output = self.get_standard_deblock(concat0, kernel_size=[4, 4], stride=[2, 2], leak=0.0, filters=[self.output_ch_size],
                                               training=training, is_bnorm=False, is_activate=True)

        '''
        FULLY-CONNECTION PART
        '''
        with tf.variable_scope('tanh'):
            output = tf.tanh(output)

        if self.learning_type == 'residual':
            with tf.variable_scope('add'):
                output = tf.add(output, input)

        return output


    '''
    DISCRIMINATOR STRUCTURE
    '''
    def get_discriminator(self, input, training):
        '''
        ENCODER PART
        '''
        stg = 0
        with tf.variable_scope('enc_{}'.format(stg)):
            enc0 = self.get_standard_block(input, kernel_size=[4, 4], stride=2, leak=0.2, filters=[64], training=training, is_bnorm=False, is_activate=True)

        stg = 1
        with tf.variable_scope('enc_{}'.format(stg)):
            enc1 = self.get_standard_block(enc0, kernel_size=[4, 4], stride=2, leak=0.2, filters=[64 * 2], training=training, is_bnorm=True, is_activate=True)

        stg = 2
        with tf.variable_scope('enc_{}'.format(stg)):
            enc2 = self.get_standard_block(enc1, kernel_size=[4, 4], stride=2, leak=0.2, filters=[64 * 4], training=training, is_bnorm=True, is_activate=True)

        stg = 3
        with tf.variable_scope('enc_{}'.format(stg)):
            enc3 = self.get_standard_block(enc2, kernel_size=[4, 4], stride=1, leak=0.2, filters=[64 * 8], training=training, is_bnorm=True, is_activate=True)

        stg = 4
        with tf.variable_scope('enc_{}'.format(stg)):
            enc4 = self.get_standard_block(enc3, kernel_size=[4, 4], stride=1, leak=0.2, filters=[1], training=training, is_bnorm=False, is_activate=False)

        with tf.variable_scope('sigmoid'):
            output = tf.sigmoid(enc4)

        return output


    '''
    GET STANDARD BLOCK: CONV -> BNORM -> RELU 
    '''
    def get_standard_block(self, input, filters, kernel_size=[3, 3], stride=[1, 1], padding='same', leak=0.0, training=True, is_bnorm=True, is_activate=True):
        output = input

        for i, f in enumerate(filters):
            with tf.variable_scope('conv_{}'.format(i)):
                output = tf.layers.conv2d(output,
                                          filters=f,
                                          kernel_size=kernel_size,
                                          strides=stride,
                                          padding=padding,
                                          kernel_initializer=self.initializer,
                                          kernel_regularizer=self.regularizer,
                                          activation=None)

            if is_bnorm:
                with tf.variable_scope('inorm_{}'.format(i)):
                    output = tf.contrib.layers.instance_norm(output)

            if is_activate:
                with tf.variable_scope('relu_{}'.format(i)):
                    if leak == 0:
                        output = tf.nn.relu(output)
                    else:
                        output = tf.nn.leaky_relu(output, leak)

        return output


    '''
    GET STANDARD BLOCK: CONV -> BNORM -> RELU 
    '''
    def get_standard_deblock(self, input, filters, kernel_size=[3, 3], stride=1, padding='same', leak=0.0, training=True, is_bnorm=True, is_activate=True):
        output = input

        for i, f in enumerate(filters):
            if is_activate:
                with tf.variable_scope('relu_{}'.format(i)):
                    if leak == 0:
                        output = tf.nn.relu(output)
                    else:
                        output = tf.nn.leaky_relu(output, leak)

            with tf.variable_scope('conv_{}'.format(i)):
                output = tf.layers.conv2d_transpose(output,
                                                    filters=f,
                                                    kernel_size=kernel_size,
                                                    strides=stride,
                                                    padding=padding,
                                                    kernel_initializer=self.initializer,
                                                    kernel_regularizer=self.regularizer,
                                                    activation=None)

            if is_bnorm:

                with tf.variable_scope('inorm_{}'.format(i)):
                    output = tf.contrib.layers.instance_norm(output)

        return output

    def get_residual_block(self, input, filters, kernel_size=[3, 3], stride=1, padding='same', leak=0.0, training=True, is_bnorm=True, is_activate=True):
        output = input

        for i, f in enumerate(filters):
            with tf.variable_scope('conv_{}'.format(i)):
                output = tf.layers.conv2d(output,
                                          filters=f,
                                          kernel_size=kernel_size,
                                          strides=stride,
                                          padding=padding,
                                          kernel_initializer=self.initializer,
                                          kernel_regularizer=self.regularizer,
                                          activation=None)
            if is_bnorm:
                with tf.variable_scope('bnorm_{}'.format(i)):
                    output = tf.layers.batch_normalization(output,
                                                           training=training)

            if i == len(filters) - 1:
                with tf.variable_scope('add'.format(i)):
                    output = tf.add(output, input)

            if is_activate:
                with tf.variable_scope('relu_{}'.format(i)):
                    if leak == 0:
                        output = tf.nn.relu(output)
                    else:
                        output = tf.nn.lrelu(output, leak)

        return output


    '''
    GET SQUENTIAL CONV-BNORM-RELU LAYERS
    '''
    def get_conv2d(self, input, filters, kernel_size=[3, 3], stride=1, padding='same', name='conv'):
        with tf.variable_scope(name):
            output = tf.layers.conv2d(input,
                                      filters=filters,
                                      kernel_size=kernel_size,
                                      strides=stride,
                                      padding=padding,
                                      kernel_initializer=self.initializer,
                                      kernel_regularizer=self.regularizer,
                                      activation=None)
        return output

    def get_bnorm2d(self, input, training=True, name='bnorm'):
        with tf.variable_scope(name):
            output = tf.layers.batch_normalization(input,
                                                   training=training)
        return output

    def get_inorm2d(self, input, name='inorm', epsilon=1e-5):
        """Instance Normalization.

        See Ulyanov, D., Vedaldi, A., & Lempitsky, V. (2016).
        Instance Normalization: The Missing Ingredient for Fast Stylization,
        Retrieved from http://arxiv.org/abs/1607.08022

        Parameters
        ----------
        x : TYPE
            Description
        epsilon : float, optional
            Description

        Returns
        -------
        TYPE
            Description
        """
        with tf.variable_scope(name):
            mean, var = tf.nn.moments(input, [1, 2], keep_dims=True)
            scale = tf.get_variable(
                name='scale',
                shape=[input.get_shape()[-1]],
                initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
            offset = tf.get_variable(
                name='offset',
                shape=[input.get_shape()[-1]],
                initializer=tf.constant_initializer(0.0))
            output = scale * tf.div(input - mean, tf.sqrt(var + epsilon)) + offset
            return output

    def get_relu(self, input, name='relu'):
        with tf.variable_scope(name):
            output = tf.nn.relu(input)
        return output

    def get_lrelu(self, input, leak=0.2, name='lrelu'):
        with tf.variable_scope(name):
            output = tf.nn.leaky_relu(input, leak)
        return output

    '''
    GET POOLING LAYERS such as AVERAGE POOLING and MAX POOLING
    '''
    def get_pool2d(self, input, pool_size=2, strides=2, padding='same', type='avg', name='pool_{}'):
        with tf.variable_scope(name.format(type)):
            if type == 'avg':
                output = tf.layers.average_pooling2d(input,
                                                     pool_size=pool_size,
                                                     strides=strides,
                                                     padding=padding)
            elif type == 'max':
                output = tf.layers.max_pooling2d(input,
                                                 pool_size=pool_size,
                                                 strides=strides,
                                                 padding=padding)
        return output

    '''
    GET UN-POOLING LAYERS such as AVERAGE POOLING and UNSAMPLING using CONVT LAYER
    '''
    def get_unpool2d(self, input, pool_size=2, strides=2, padding='same', type='avg', name='unpool_{}'):
        with tf.variable_scope(name.format(type)):
            if type == 'avg':
                output = 1.0/(pool_size**2)*tf.cast(tf.image.resize_nearest_neighbor(input,
                                                                                  size=[input.shape.dims[1] * pool_size,
                                                                                        input.shape.dims[2] * pool_size]),
                                                    dtype=self.data_type)
            elif type == 'convt':
                output = tf.layers.conv2d_transpose(input,
                                                    filters=input.shape.dims[3],
                                                    kernel_size=pool_size,
                                                    strides=strides,
                                                    padding=padding,
                                                    kernel_initializer=self.initializer,
                                                    kernel_regularizer=self.regularizer)
        return output

    '''
    GET CONCAT LAYERS
    '''
    def get_concat(self, inputs, axis=3):
        output = tf.concat(inputs, axis=axis)
        return output

    '''
    GET FFT LAYERS
    '''
    def get_fft(self, inputs):
        # H W C B
        output = tf.transpose(inputs, perm=[1, 2, 3, 0])
        sz = tf.shape(inputs)

        output = tf.complex(output[:, :, 0:sz[2]/2, :], output[:, :, sz[2]/2:, :])

        # IFFTSHIFT: 2D
        output = tf.manip.roll(output, shift=[tf.ceil(sz[0]/2), tf.ceil(sz[1]/2)], axis=[0, 1])
        # FFT: 2D
        output = tf.spectral.fft2d(output)
        # FFTSHIFT: 2D
        output = tf.manip.roll(output, shift=[tf.floor(sz[0]/2), tf.floor(sz[1]/2)], axis=[0, 1])

        # COMPLEX to [REAL, IMAG]
        output = tf.concat([tf.real(output), tf.imag(output)], axis=2)

        # B H W C
        output = tf.transpose(output, perm=[3, 0, 1, 2])

        return output

    '''
    GET IFFT LAYERS
    '''
    def get_ifft(self, inputs):
        # H W C B
        output = tf.transpose(inputs, perm=[1, 2, 3, 0])
        sz = tf.shape(inputs)

        # [REAL, IMAG] to COMPLEX
        output = tf.complex(output[:, :, :(sz[2]/2), :], output[:, :, (sz[2]/2):, :])

        # FFTSHIFT: 2D
        output = tf.manip.roll(output, shift=[tf.floor(sz[0] / 2), tf.floor(sz[1] / 2)], axis=[0, 1])
        # IFFT: 2D
        output = tf.spectral.ifft2d(output)
        # IFFTSHIFT: 2D
        output = tf.manip.roll(output, shift=[tf.ceil(sz[0] / 2), tf.ceil(sz[1] / 2)], axis=[0, 1])

        # COMPLEX to [REAL, IMAG]
        output = tf.concat([tf.real(output), tf.imag(output)], axis=2)

        # B H W C
        output = tf.transpose(output, perm=[3, 0, 1, 2])

        return output