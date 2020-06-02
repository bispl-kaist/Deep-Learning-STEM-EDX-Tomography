#!/usr/bin/env bash

python main.py \
--mode train \
--scope sr \
--checkpoint_id -1 \
--output_dir ./result/test/sr \
--input_dir ./data/train/sr \
--epoch_num 1000 \
--decay_step 1000 \
--patch_y_size 256 \
--patch_x_size 256 \
--patch_ch_size 3 \
--load_y_size 286 \
--load_x_size 286 \
--load_ch_size 3 \
--beta 0.9 \
--device 0 \
--network_type 'cgan'