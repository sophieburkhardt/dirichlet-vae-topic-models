# NEWS - Feb. 20, 2023

Check out the Pytorch version of the Dirichlet Variational Autoencoder, implemented with Recjection Sampling Variational Inference, available at [https://github.com/mayanknagda/neural-topic-models](https://github.com/mayanknagda/neural-topic-models).

Another Pytorch version of the DVAE using [pathwise gradients](http://proceedings.mlr.press/v80/jankowiak18a.html) is available here: [https://github.com/ahoho/topics](https://github.com/ahoho/topics).

# Dirichlet Variational Autoencoders

Implementation of different [Dirichlet Variational Autoencoders](https://www.datamining.informatik.uni-mainz.de/files/2019/07/Burkhardt_Kramer_DVAE.pdf). Accepted in JMLR 2019

Implements the following methods
- [Dirichlet-Autoencoder with "implicit gradients"](https://arxiv.org/pdf/1805.08498.pdf)
- [Dirichlet-Autoencoder with RSVI](https://arxiv.org/abs/1610.05683)
- [Dirichlet-Autoencoder with inverse CDF](https://arxiv.org/abs/1901.02739)
- [Dirichlet-Autoencoder with Weibull distribution](https://arxiv.org/abs/1803.01328)

# Running the Code # 

## RSVI
`python3 nvdm_dirichlet_rsvi.py 2`

## Implicit Gradients

`python3 nvdm_dirichlet_implicitGradients.py 4`

Only with current Tensorflow version. Need to install tensorflow-probability

## Inverse CDF

`python3 nvdm_dirichlet_invCDF.py 4`

## Weibull

`python3 nvdm_dirichlet_weibull.py 4`

# Dataset format

See example dataset in data folder
