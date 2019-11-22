#!/bin/bash
export OMP_NUM_THREADS=1


filename=$(date +"%Y%m%d-%H%M%S-%N")".log"

export CUDA_VISIBLE_DEVICES=0
python3 main.py   \
  --phase train \
  --method regression \
  --optimizer adam \
  --lrG 0.001\
  --lrG-lower-bound 0.00002\
  --lrG-half-life 400 \
  --max-color-index 64.0 \
  --min-color-index 0.0 \
  --no-display-img 8 \
  --dataroot regression_data/1st_atom \
  --dataroot-val regression_data/1st_atom \
  --imageSize 128 --nc 1 --workers 16\
  --batchSize  32 \
  --nepochs 299 \
  --nsave 1 \
  --model TightFrameUnet \
  --filtSize 3 \
  --residual \
  --ngpus 1 \
  --bn \
  --no-random 30 \
  --experiment samples/patch128_batch32_residual_no_random_30 \
  2>&1 | tee logs/patch128_batch32_residual_no_random_30_$filename





