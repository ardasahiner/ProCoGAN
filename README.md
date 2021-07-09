# ProCoGAN

Implementation for progressive training of convex Wasserstein GANs. Built on theory that demonstrates that GANs with linear generators and two-layer quadratic-activation neural network discriminators are convex programs and can be solved in closed-form, and motivated by the success of progressive training of GANs, we show that the convex formulation improves upon baseline heuristic stochastic Gradient Descent-Ascent (GDA), which is typically used in practice. This repository includes implementations of the convex form and the baseline on the CelebA faces dataset. 
