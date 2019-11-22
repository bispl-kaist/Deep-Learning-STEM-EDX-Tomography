#!/bin/bash
export OMP_NUM_THREADS=1

filename=$(date +"%Y%m%d-%H%M%S-%N")".log"
export CUDA_VISIBLE_DEVICES=1
python3 main.py   \
  --phase test \
  --method regression \
  --optimizer adam \
  --lrG 0.0001\
  --lrG-lower-bound 0.00002\
  --lrG-half-life 100 \
  --max-color-index 128.0 \
  --min-color-index 0.0 \
  --no-display-img 1 \
  --dataroot-val attention_data/1st_atom \
  --imageSize 256 --nc 1 --workers 16\
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
  --load-pretrained-generator-path samples/patch128_batch1_residual_bn_fine_3layer_proj_loss/gen-2\
  --experiment samples/patch128_batch1_residual_bn_fine_3layer_proj_loss_test \
  2>&1 | tee logs/patch128_batch1_residual_bn_fine_3layer_proj_loss_$filename

