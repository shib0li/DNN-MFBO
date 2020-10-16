#!/bin/sh

python main.py --domain Park1 \
               --inits 20,2 \
               --epochs 5000 \
               --maxiter 40 \
               --activation relu \
               --hlayers_width 128,128 \
               --hlayers_depth 2,2 \
               --klayers 32,32 \
               --trials 1  \
                  