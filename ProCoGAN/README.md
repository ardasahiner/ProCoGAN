# ProCoGAN

Use for running the Progressive Convex GAN as described in Figure 2 of our paper. This leverages the closed-form solution for linear generators and quadratic-activation two-layer discriminators for progressively generating CelebA-style face images. The jupyter notebook contains all code required to run this. Requirements for running are numpy, Pytorch, scipy, scikit-image, datetime, pillow, and matplotlib. 

To run, it is required to download CelebA align & cropped images [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), as well as attributes annotations if fitting faces with a particular attribute is desired. 
