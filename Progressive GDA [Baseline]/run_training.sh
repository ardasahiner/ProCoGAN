#!/bin/bash

# training parameters

dir=/

CUDA_VISIBLE_DEVICES=0 python train.py PGAN -c config_celeba_cropped_fr_alpha05_LQ.json -d $dir

python eval.py visualization -n default -m PGAN -d $dir

python models/eval/fid_hist_match.py --dir $dir

