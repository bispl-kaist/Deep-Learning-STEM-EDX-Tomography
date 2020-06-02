#!/usr/bin/env bash

python main.py \
--mode test \
--scope sr \
--checkpoint_id -1 \
--output_dir ./result/test/sr/1 \
--input_dir ./data/test/sr/1 \
--epoch_num 1000 \
--decay_step 1000 \
--patch_y_size 256 \
--patch_x_size 256 \
--patch_ch_size 3 \
--load_y_size 256 \
--load_x_size 256 \
--load_ch_size 3 \
--beta 0.9 \
--device 0 \
--network_type 'cgan'


python main.py \
--mode test \
--scope sr \
--checkpoint_id -1 \
--output_dir ./result/test/sr/2 \
--input_dir ./data/test/sr/2 \
--epoch_num 1000 \
--decay_step 1000 \
--patch_y_size 256 \
--patch_x_size 256 \
--patch_ch_size 3 \
--load_y_size 256 \
--load_x_size 256 \
--load_ch_size 3 \
--beta 0.9 \
--device 0 \
--network_type 'cgan'


python main.py \
--mode test \
--scope sr \
--checkpoint_id -1 \
--output_dir ./result/test/sr/3 \
--input_dir ./data/test/sr/3 \
--epoch_num 1000 \
--decay_step 1000 \
--patch_y_size 256 \
--patch_x_size 256 \
--patch_ch_size 3 \
--load_y_size 256 \
--load_x_size 256 \
--load_ch_size 3 \
--beta 0.9 \
--device 0 \
--network_type 'cgan'