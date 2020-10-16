#!/bin/sh

python main.py --domain Branin \
               --inits 20,10,2 \
               --epochs 5000 \
               --maxiter 40 \
               --activation relu \
               --hlayers_width 547,229,19 \
               --hlayers_depth 2,3,3 \
               --klayers 43,397,195 \
               --trials 5  \
                  