#!/usr/bin/env bash

python main_sait.py \
--mode test  \
--scope sait_recon_cgan_avg5_ep30  \
--checkpoint_id -1  \
--output_dir ./test/sait_recon_full/cgan/avg5/epoch30/train1000/1  \
--input_dir ./data/sait_recon_full/avg5/epoch30/1  \
--epoch_num 1000  \
--decay_step 1000  \
--patch_y_size 256  \
--patch_x_size 256  \
--patch_ch_size 3  \
--load_y_size 286  \
--load_x_size 286  \
--load_ch_size 3  \
--beta 0.9  \
--device 0  \
--network_type 'cgan'
