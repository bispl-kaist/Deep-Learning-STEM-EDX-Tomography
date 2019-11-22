#!/bin/bash
export OMP_NUM_THREADS=1


filename=$(date +"%Y%m%d-%H%M%S-%N")".log"

export CUDA_VISIBLE_DEVICES=0
python3 main.py   \
  --phase test \
  --method regression \
  --max-color-index 64.0 \
  --min-color-index 0.0 \
  --no-display-img 1 \
  --dataroot regression_data/1st_atom \
  --dataroot-val regression_data/1st_atom \
  --imageSize 128 --nc 1 --workers 16\
  --batchSize  1 \
  --nepochs 299 \
  --nsave 1 \
  --model TightFrameUnet \
  --filtSize 3 \
  --residual \
  --ngpus 1 \
  --bn \
  --no-random 30 \
  --load-pretrained-generator-path samples/patch128_batch32_residual_no_random_30/gen-299 \
  --experiment samples/patch128_batch32_residual_no_random_30_test\
  2>&1 | tee logs/patch128_batch32_residual_no_random_30_test_$filename





