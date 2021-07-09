import numpy as np
import torch
#import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from scipy.linalg import hadamard
import copy
from PIL import Image
import os
import matplotlib.pyplot as plt
import time
import scipy
import argparse
import math
import torch.nn.functional as F
import torchvision
import torch.nn as nn
from torchvision import models

#from mpl_toolkits.axes_grid1 import ImageGrid
from skimage import exposure
from skimage.exposure import match_histograms
from scipy.sparse import linalg
from datetime import datetime

def composition(im_path, IM_SIZE, cropped=True):
    cx = 89
    cy = 121
    
    # center crop to 128 x 128, then resize to relevant dimension
    with Image.open(im_path) as im1:
        if cropped:
            im1 = im1.crop((cx-64, cy-64, cx+64, cy+64))
        return im1.resize((IM_SIZE, IM_SIZE))

## FID stuff 

class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False):
        
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        
        inception = models.inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps
        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)
        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x,
                              size=(299, 299),
                              mode='bilinear',
                              align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp
    
def calculate_activation_statistics(images,model,batch_size=100, dims=2048,
                    cuda=False):
    model.eval()
    act=np.empty((len(images), dims))
    nBatches = len(images)//batch_size
    
    for i in range(nBatches):
        batch = images[i*batch_size:(i+1)*batch_size]
        if cuda:
            batch=batch.cuda()

        pred = model(batch)[0]

            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        act[i*batch_size:(i+1)*batch_size]= pred.cpu().data.numpy().reshape(pred.size(0), -1)
    
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_fretchet(images_real,images_fake, batch_size=10):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx])
    model=model.cuda()
    
    images_real = torch.from_numpy(images_real).float().permute(0, 3, 1, 2)
    images_fake = torch.from_numpy(images_fake).float().permute(0, 3, 1, 2)
    
    mu_1,std_1=calculate_activation_statistics(images_real,model,batch_size=batch_size,cuda=True)
    mu_2 ,std_2=calculate_activation_statistics(images_fake,model,batch_size=batch_size,cuda=True)
    
    """get fretched distance"""
    fid_value = calculate_frechet_distance(mu_1, std_1, mu_2, std_2)
    print('FID score', fid_value)
    np.save(exp_dir+'fid_value',fid_value)
    return fid_value
    
parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='',help = 'Experiment directory')
args = parser.parse_args()
root = ""
exp_dir = str(args.dir)+'/default/'
display_img = np.load(exp_dir+'default_s4_i15000_fullavgval_ims.npy')
final_resolution = 32
curr_res = final_resolution
cropped = False
print('data')
images_path =np.array([ os.path.join(root, item)  for item in os.listdir(root)])
lazy_arrays = [composition(fn, final_resolution, cropped=cropped) for fn in images_path]
curr_arrs = lazy_arrays
image_data = np.stack(curr_arrs)
celebA= image_data/255

display_img = np.transpose(display_img,(0,2,3,1))

display_img = match_histograms(display_img, celebA, multichannel=True)
print('saving images')
dirname = exp_dir+'/'+str(datetime.now())
os.mkdir(dirname)
for i, im in enumerate(display_img):
     plt.imsave(os.path.join(dirname, str(i)+'.png'), im)

calculate_fretchet(celebA,display_img, batch_size=50)