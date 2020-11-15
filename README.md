# VBA-python

Variational Bayesian (VB) analysis provides a technique for fully Bayesian parameter estimation in nonlinear models formulated with ordinary differential equations (ODEs). This means that all unknown parameters are considered to be continuous random variables following a parametric probability density function (PDF). In addition to the unknown parameters, the VB method provides a lower bound to the marginal likelihood, also known as model evidence, which can be used for model selection. The VB method is fully deterministic thereby using semi-analytical approximations to the true posterior distributions and provides an efficient alternative to Markov Chain Monte Carlo approaches. The VB approach for the inversion of stochastic ODE models is implemented in an open-source MATLAB toolbox (https://mbb-team.github.io/VBA-toolbox/). 

The purpose of the here described software package is to provide an implementation of VB method in Python. The main difference to the MATLAB toolbox is that we only allow the inversion of deterministic ODE models. Additionally, the focus lies on the inversion of models used in physiology. 

# License

This software is distributed under a GNU open-source licence. You are free to download and modify it.

M. Eichenlaub 15/11/2020
