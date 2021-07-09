# Non-Convex Progressive GDA 

This repository contains the scripts used to run the progressive GDA experiments in the submission. This repository was build off of the Pytorch GAN Zoo (https://github.com/facebookresearch/pytorch_GAN_zoo) toolbox, and general information about this code base is also provided in 'README_proggan.md'. This file includes information regarding the requirements, configuring the datasets used and so on.

To reproduce the Progressive GDA experiments, one can run:

python train.py PGAN -c config_celeba_cropped_fr_alpha05_LQ.json -d $dir

where 'dir' is the results directory. The config given above replicates the settings used in the paper, though in the config file the pathDB variable must be specified to point to the directory of the CelebA dataset. To run evaluation, and generate additional images from the trained networks, one can run:

python eval.py visualization -n default -m PGAN -d $dir

where 'default' refers to the name of training. After evaluation, to perform histogram matching and FID calculation, one can run:

python models/eval/fid_hist_match.py --dir $dir

These commands are given together in the file "run_training.sh".

There are several parameters that are currently hardcoded, such as the generator regularization. In addition, because we demean the data before training and add it back in evaluation, means at each resolution are needed to be stored and are provided in our repository.
