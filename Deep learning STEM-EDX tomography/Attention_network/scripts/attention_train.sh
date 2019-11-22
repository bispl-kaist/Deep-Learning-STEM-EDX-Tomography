#!/bin/bash
export OMP_NUM_THREADS=1

filename=$(date +"%Y%m%d-%H%M%S-%N")".log"
export CUDA_VISIBLE_DEVICES=1
python3 main.py   \
  --phase train \
  --method regression \
  --optimizer adam \
  --lrG 0.0001\
  --lrG-lower-bound 0.00002\
  --lrG-half-life 100 \
  --max-color-index 128.0 \
  --min-color-index 0.0 \
  --no-display-img 1 \
  --dataroot attention_data/1st_atom \
  --dataroot-val attention_data/1st_atom \
  --imageSize 128 --nc 1 --workers 16\
  --batchSize  1 \
  --nepochs 399 \
  --nsave 1 \
  --filtSize 3 \
  --residual \
  --ngpus 1 \
  --bn \
  --proj \
  --avg \
  --no-random 30 \
  --lambda-random 0.1 \
  --experiment samples/patch128_batch1_residual_bn_fine_3layer_proj_loss \
  2>&1 | tee logs/patch128_batch1_residual_bn_fine_3layer_proj_loss_$filename

