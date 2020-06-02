#!/usr/bin/env bash

python main.py \
--mode test \
--scope denoising \
--checkpoint_id -1 \
--output_dir ./result/denoising \
--input_dir ./data/denoising \
--epoch_num 30 \
--decay_step 30 \
--patch_y_size 256 \
--patch_x_size 256 \
--patch_ch_size 3  \
--load_y_size 256 \
--load_x_size 256 \
--load_ch_size 3 \
--beta 0.9 \
--device 0 \
--network_type 'cgan'