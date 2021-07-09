# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import torch.nn.functional as F

from .custom_layers import EqualizedConv2d, EqualizedLinear,\
    NormalizationLayer, Upscale2d, Polynomial, Quadratic, Linear
from ..utils.utils import num_flat_features
from.mini_batch_stddev_module import miniBatchStdDev


class GNet(nn.Module):

    def __init__(self,
                 dimLatent,
                 depthScale0,
                 initBiasToZero=True,
                 leakyReluLeak=0.0,
                 normalization=False,
                 generationActivation=None,
                 dimOutput=3,
                 equalizedlR=True,
                 activation = "Linear",
                 num_conv_layers = 2,
                 train_prev_layers = False):
        r"""
        Build a generator for a progressive GAN model

        Args:

            - dimLatent (int): dimension of the latent vector
            - depthScale0 (int): depth of the lowest resolution scales
            - initBiasToZero (bool): should we set the bias to zero when a
                                    new scale is added
            - leakyReluLeak (float): leakyness of the leaky relu activation
                                    function
            - normalization (bool): normalize the input latent vector
            - generationActivation (function): activation function of the last
                                               layer (RGB layer). If None, then
                                               the identity is used
            - dimOutput (int): dimension of the output image. 3 -> RGB, 1 ->
                               grey levels
            - equalizedlR (bool): set to true to initiualize the layers with
                                  N(0,1) and apply He's constant at runtime

        """
        super(GNet, self).__init__()
        
        self.activation = activation
        self.num_conv_layers = num_conv_layers
        self.train_prev_layers = train_prev_layers

        self.equalizedlR = equalizedlR
        self.initBiasToZero = initBiasToZero

        # Initalize the scales
        self.scalesDepth = [depthScale0]
        self.scaleLayers = nn.ModuleList()
        self.toRGBLayers = nn.ModuleList()

        # Initialize the scale 0
        self.initFormatLayer(dimLatent)
        self.dimOutput = dimOutput
        self.groupScale0 = nn.ModuleList()
        
        self.groupScale0.append(EqualizedLinear(depthScale0 * 16,
                                                   depthScale0,
                                                   equalized=equalizedlR,
                                                   initBiasToZero=initBiasToZero))

        self.toRGBLayers.append(nn.ModuleList())

        # Initalize the upscaling parameters
        # alpha : when a new scale is added to the network, the previous
        # layer is smoothly merged with the output in the first stages of
        # the training
        self.alpha = 0
        
        # Leaky relu activation
        if self.activation == 'LeakyReLU':
            self.leakyRelu = torch.nn.LeakyReLU(leakyReluLeak)
        elif self.activation == 'Polynomial':
            self.leakyRelu = Polynomial()
        elif self.activation == 'Quadratic':
            self.leakyRelu = Quadratic()
        elif self.activation == 'LQ':
            self.leakyRelu = Linear()

        # normalization
        self.normalizationLayer = None
        if normalization:
            self.normalizationLayer = NormalizationLayer()

        # Last layer activation function
        self.generationActivation = generationActivation
        self.depthScale0 = depthScale0


    def initFormatLayer(self, dimLatentVector):
        r"""
        The format layer represents the first weights applied to the latent
        vector. It converts a 1xdimLatent input into a 4 x 4 xscalesDepth[0]
        layer.
        """

        self.dimLatent = dimLatentVector
        
        self.formatLayer = EqualizedLinear(self.dimLatent,
                                           16 * self.scalesDepth[0],
                                           equalized=self.equalizedlR,
                                           initBiasToZero=self.initBiasToZero)

    def getOutputSize(self):
        r"""
        Get the size of the generated image.
        """
        side = 4 * (2**(len(self.toRGBLayers) - 1))
        return (side, side)

    def addScale(self, depthNewScale):
        r"""
        Add a new scale to the model. Increasing the output resolution by
        a factor 2

        Args:
            - depthNewScale (int): depth of each conv layer of the new scale
        """

        if self.train_prev_layers == False:
            if len(self.scaleLayers) == 0:
                prev_layer = self.formatLayer
            else:
                prev_layer = self.scaleLayers[-1]
            for param in prev_layer.parameters():
                param.requires_grad = False
            prev_layer.eval()
        
        depthLastScale = self.scalesDepth[-1]

        self.scalesDepth.append(depthNewScale)

        self.scaleLayers.append(nn.ModuleList())
        
        self.scaleLayers[-1].append(EqualizedLinear(depthNewScale * 4,
                                               depthNewScale * 16,
                                               equalized=self.equalizedlR,
                                               initBiasToZero=self.initBiasToZero))
        self.toRGBLayers.append(nn.ModuleList())

    def setNewAlpha(self, alpha):
        r"""
        Update the value of the merging factor alpha

        Args:

            - alpha (float): merging factor, must be in [0, 1]
        """

        if alpha < 0 or alpha > 1:
            raise ValueError("alpha must be in [0,1]")

        if not self.toRGBLayers:
            raise AttributeError("Can't set an alpha layer if only the scale 0"
                                 "is defined")

        self.alpha = alpha

    def forward(self, x):

        x = x.view(-1, num_flat_features(x))
        # format layer
        x = self.leakyRelu(self.formatLayer(x))
                
        # Upper scales
        for scale, layerGroup in enumerate(self.scaleLayers, 0):
            x = x.view(-1, num_flat_features(x))
            for convLayer in layerGroup:
                x = self.leakyRelu(convLayer(x))
        
        d = len(self.scaleLayers)
        x = x.view(x.size()[0], -1, 4*(2**d), 4*(2**d))
        return x


class DNet(nn.Module):

    def __init__(self,
                 depthScale0,
                 initBiasToZero=True,
                 leakyReluLeak=0.0,
                 sizeDecisionLayer=1,
                 miniBatchNormalization=False,
                 dimInput=3,
                 equalizedlR=True,
                 activation = "Quadratic",
                 num_conv_layers = 2,
                 train_prev_layers = False):
        r"""
        Build a discriminator for a progressive GAN model

        Args:

            - depthScale0 (int): depth of the lowest resolution scales
            - initBiasToZero (bool): should we set the bias to zero when a
                                    new scale is added
            - leakyReluLeak (float): leakyness of the leaky relu activation
                                    function
            - decisionActivation: activation function of the decision layer. If
                                  None it will be the identity function.
                                  For the training stage, it's advised to set
                                  this parameter to None and handle the
                                  activation function in the loss criterion.
            - sizeDecisionLayer: size of the decision layer. Will typically be
                                 greater than 2 when ACGAN is involved
            - miniBatchNormalization: do we apply the mini-batch normalization
                                      at the last scale ?
            - dimInput (int): 3 (RGB input), 1 (grey-scale input)
        """
        super(DNet, self).__init__()
        
        self.activation = activation
        self.num_conv_layers = num_conv_layers
        self.train_prev_layers = train_prev_layers

        # Initialization paramneters
        self.initBiasToZero = initBiasToZero
        self.equalizedlR = equalizedlR
        self.dimInput = dimInput

        # Initalize the scales
        self.scalesDepth = [depthScale0]
        self.scaleLayers = nn.ModuleList()
        self.fromRGBLayers = nn.ModuleList()

        self.mergeLayers = nn.ModuleList()

        # Initialize the last layer
        self.initDecisionLayer(sizeDecisionLayer)

        # Layer 0
        self.groupScaleZero = nn.ModuleList()
        self.fromRGBLayers.append(EqualizedConv2d(dimInput, depthScale0, 1,
                                                  equalized=equalizedlR,
                                                  initBiasToZero=initBiasToZero))

        # Minibatch standard deviation
        dimEntryScale0 = depthScale0
        if miniBatchNormalization:
            dimEntryScale0 += 1

        self.miniBatchNormalization = miniBatchNormalization
        self.groupScaleZero.append(EqualizedConv2d(dimEntryScale0, depthScale0,
                                                   3, padding=1,
                                                   equalized=equalizedlR,
                                                   initBiasToZero=initBiasToZero))

        self.groupScaleZero.append(EqualizedLinear(depthScale0 * 16,
                                                   depthScale0 * 64,
                                                   equalized=equalizedlR,
                                                   bias=False,
                                                initBiasToZero=False))

        # Initalize the upscaling parameters
        self.alpha = 0

        # Leaky relu activation
        if self.activation == 'LeakyReLU':
            self.leakyRelu = torch.nn.LeakyReLU(leakyReluLeak)
        elif self.activation == 'Polynomial':
            self.leakyRelu = Polynomial()
        elif self.activation == 'LQ':
            self.leakyRelu = Quadratic()

    def addScale(self, depthNewScale):
        if depthNewScale == 768:
            hiddendim = int(depthNewScale/4)
        else:
            hiddendim = depthNewScale
            
        self.groupScaleZero[1] = EqualizedLinear(depthNewScale * 16,
                                                   hiddendim*16,
                                                   equalized=self.equalizedlR,bias=False,
                                                initBiasToZero=False)
        self.decisionLayer = EqualizedLinear(hiddendim*16,
                                             1,
                                             equalized=self.equalizedlR,bias=False,
                                             initBiasToZero=False)
    
    def setNewAlpha(self, alpha):
        r"""
        Update the value of the merging factor alpha

        Args:

            - alpha (float): merging factor, must be in [0, 1]
        """

        if alpha < 0 or alpha > 1:
            raise ValueError("alpha must be in [0,1]")

        if not self.fromRGBLayers:
            raise AttributeError("Can't set an alpha layer if only the scale 0"
                                 "is defined")

        self.alpha = alpha

    def initDecisionLayer(self, sizeDecisionLayer):

        self.decisionLayer = EqualizedLinear(self.scalesDepth[0]*64,
                                             sizeDecisionLayer,
                                             equalized=self.equalizedlR,
                                             bias=False,
                                            initBiasToZero=False)
        
    def forward(self, x, getFeature = False):

        x = x.view(-1, num_flat_features(x))
        x = self.leakyRelu(self.groupScaleZero[1](x))
        
        out = self.decisionLayer(x)

        if not getFeature:
            return out

        return out, x
