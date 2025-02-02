# Guiding-Neural-Collapse

This is the code for the [paper](https://arxiv.org/abs/2411.01248) "Guiding Neural Collapse: Optimising Towards the Nearest Equiangular Tight Frame".

Neural Information Processing Systems (NeurIPS), 2024

## Introduction

- We introduce the notion of the nearest simplex ETF geometry given the penultimate layer features. Instead of selecting a predetermined simplex ETF (canonical or random), we implicitly fix the classifier as the solution to a Riemannian optimisation problem.
- To establish end-to-end learning, we encapsulate the Riemannian optimisation problem of determining the nearest simplex ETF geometry within a declarative node. This allows for efficient backpropagation throughout the network.
- We demonstrate that our method achieves an optimal neural collapse solution more rapidly compared to fixed simplex ETF methods or conventional training approaches, where a learned linear classifier is employed. Additionally, our method ensures training stability by markedly reducing variance in network performance.

## Environment
- python
- numpy
- scipy
- matplotlib
- pytorch
- torchvision
- jax

## Datasets

By default, the code assumes the datasets for MNIST and CIFAR10 are stored under `~/data/`. If the datasets are not there, they will be automatically downloaded from `torchvision.datasets`. User may change this default location of datasets in `args.py` through the argument `--data_dir`.


## Overview of the ETF Geometry Optimisation Module


The file `ddn_modules.py` provides a reference implementation for computing the closest Equiangular Tight Frame (ETF) geometry using Riemannian optimisation. It is designed for research purposes and serves as a clear, illustrative example rather than a fully optimised production solution.

### Key Components

- **`ClosestETFGeometryFcn`**  
  A custom `torch.autograd.Function` that defines both the forward and backward passes for computing the closest ETF geometry.  
  - **Forward Pass:**  
    - Performs Riemannian optimisation on the Stiefel manifold to obtain a transformation matrix `P` that minimises a cost function.
    - The cost function comprises a term measuring the Frobenius norm difference between the transformed features and a target ETF structure, along with a proximal term.
    - On the first pass, it establishes an initial point to improve convergence.
  - **Backward Pass:**  
    - Computes gradients with respect to the input feature means, ensuring that the module integrates seamlessly into the training pipeline.
    - Utilises block-wise system solvers and handles both feasibility constraints and general linear systems.
  - **Diagnostics:**  
    - Optionally logs diagnostic information (e.g., objective value, stopping criteria, gradient norms) to help analyse the optimisation process.

- **`FeaturesMovingAverageLayer`**  
  A PyTorch module for maintaining a moving average of feature representations. It supports two update methods:
  - **Cumulative Moving Average (CMA)**
  - **Exponential Moving Average (EMA)**
  
  This layer also handles cases where some classes have insufficient samples by substituting missing values with a global mean.

- **`ClosestETFGeometryLayer`**  
  A wrapper module that integrates the custom autograd function into the network. It:
  - Initialises and updates the necessary variables such as the target ETF matrix (`M`), the initial transformation matrix (`P_init`), and the proximal term (`Prox`).
  - Facilitates the iterative refinement of the ETF structure during training.

### Hardware Support

The file conditionally imports GPU or CPU versions of the `pymanopt` modules based on the `HARDWARE` flag. This ensures compatibility and optimised performance on both GPU and CPU environments. 

### Usage Notes

- **Research Focus:**  
  This implementation is intended primarily for research and experimental use, particularly in studies related to neural collapse and classifier optimisation in deep learning.
- **Diagnostics and Logging:**  
  Optional logging can be enabled to monitor the optimisation process, providing insights into convergence and performance.
- **Optimisation Caveats:**  
  The code is not fully optimised for speed or memory usage. A more optimised version may be released in the future.

---

This module is a key component for experiments aiming to optimise neural networks towards an ETF structure, offering a practical example of how Riemannian optimisation can be integrated within deep learning frameworks.


## Folder Structure Overview

All executable scripts are located in the `scripts` folder, which also contains the datasets and network architecture definitions. The provided example uses the MNIST dataset with a ResNet18 model.

### Methods Available

There are three subfolders under `scripts`, each corresponding to a different training method:

- **`standard`**  
  Implements the classical training method where all network weights are optimized.

- **`fixed`**  
  Uses the fixed ETF method. In this approach, the final classifier weights are set to a canonical simplex ETF and remain fixed; only the remaining network weights are optimised.

- **`implicit`**  
  Contains our proposed implicit ETF method, where the network iteratively optimises towards the nearest simplex ETF for faster convergence.

### Common Python Scripts

Each method's folder includes the following scripts:

- **`train.py`**  
  Trains the network.

- **`validate_NC.py`**  
  Measures neural collapse statistics and metrics on both the training and test datasets.

- **`ETF_distance.py`**  
  Calculates the distance to neural collapse on both the training and test datasets.

- **`seed_statistics.py`** *(only for the implicit method)*  
  Accumulates seed statistics from all methods.  
  **Note:** Run this script after executing the other scripts.


## Citation 

For more technical details and full experimental results, please check our [paper](https://arxiv.org/abs/2411.01248).
If you would like to reference in your research please cite as:

```
@inproceedings{Markou:NIPS2024,
  author    = {Evan Markou and
               Thalaiyasingam Ajanthan and
               Stephen Gould},
  title     = {Guiding Neural Collapse: Optimising Towards the Nearest Simplex Equiangular Tight Frame},
  booktitle = {NeurIPS},
  year      = {2024}
}
```