#!/usr/bin/env bash

python main_sait.py  \
--mode test  \
--scope sait_cgan_avg5  \
--checkpoint_id -1  \
--output_dir ./test/sait/cgan/avg5/epoch30  \
--input_dir ./data/sait/avg5  \
--epoch_num 30  \
--decay_step 30  \
--patch_y_size 256  \
--patch_x_size 256  \
--patch_ch_size 3  \
--load_y_size 286  \
--load_x_size 286  \
--load_ch_size 3  \
--beta 0.9  \
--device 0  \
--network_type 'cgan'
