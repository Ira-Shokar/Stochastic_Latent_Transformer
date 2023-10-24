# Stochastic Latent Transformer

This repository contains the code for the paper: [Stochastic Latent Transformer: Efficient Modelling of Stochastically Forced Zonal Jet Dynamics](https://arxiv.org/)

In this work, we formulate a new neural operator by parameterizing the integral kernel directly in Fourier space, allowing for an expressive and efficient architecture. 
We perform experiments on Burgers' equation, Darcy flow, and the Navier-Stokes equation (including the turbulent regime). 
Our Fourier neural operator shows state-of-the-art performance compared to existing neural network methodologies and it is up to three orders of magnitude faster compared to traditional PDE solvers.

## Files

 - Directory `src` contains the python scripts to define and train the Stochatic Latent Transformer using [Pytorch](https://github.com/pytorch/pytorch).
          - `src/nn.py` defines the Translation Equivarient Pointwise 1D Convolution (TEPC) layers as well as the Multihead_Self_Attention operation and the Stochatic_Attention_Block.
          - `src/models.py` defines Autoencoder and Stochatic Transformer architectures.
          - `src/SLT.py` defines Stochastic_Latent_Transformer, the contains the training and validation loops, loss function and other training utilities. It also contains model loading, the hellinger_distance_3D evaluation metric and the inference function to generate new trajectories using the Stochastic Latent Transformer.
          - `src/train.py` defines model and training hyperparameters, loads training data and fits the model.
          - `src/eval.py` loads trained models, forecasts new trajectories from a set of inital condtions and plots the outputs.
          - `src/utils.py` defines utilities used during training and evaluation.
          - Directory `vae` defines training of the temporal VAE and adversarial training for comaparison.

- Directory `notebooks` contains notebooks used for evaluation.
          - `notebooks/evaluation.ipynb` loads trained models, forecasts new trajectories from a set of inital condtions and plots the outputs before some evaluative metrics.
          - `notebooks/jet_transitions.ipynb` plots PDFs for determaning properties of spontaneous transition events.

- Directory `QGF` contains the julia scripts used to generate the numerical integrations of the beta-plane turbulence model in 2D using [GeophysicalFlows.jl](https://github.com/FourierFlows/GeophysicalFlows.jl).
          - `QGF/forced_beta_SL.jl` defines the model parameters and forcing for a Quasi-geostrophic flow on a 2D periodic beta-plane and integrates using a pseudo-spectral solver .
          - `QGF/plots_SL.jl` reads the JLD2 files output from `forced_beta_SL.jl` and plots latitude-time plots of zonally-averaged zonal velocity (U) as well as the assocaited fields and saves the time evolution of U as a csv.

## Datasets
We provide the training data output from `QGF/forced_beta_SL.jl` used for training in `src/train.py` in the following Zenodo repositoy:
We provide the evaluation data output from `QGF/forced_beta_SL.jl` used for testing in `src/eval.py`, `notebooks/evaluation.ipynb` and `notebooks/jet_transitions.ipynb` in the Zenodo repositoy:

## Models
We provide the trained STL model in the form of a torchscript object, that can is evaluated using `eval.py`, `evaluation.ipynb` and `notebooks/jet_transitions.ipynb` in the Zenodo repositoy:

## Citations

```
```

