![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-2.3.0%2B-red?logo=pytorch&logoColor=white)

# All-KAN: All Kolmogorov-Arnold Networks We Know
This repository gathers all known Kolmogorov-Arnold Networks (including those I developed) from various sources. These networks are implemented for image classification on the MNIST and Fashion-MNIST datasets. I hope this collection inspires you and the broader research community to advance the development of even better KANs in the future.

Feel free to suggest or integrate your KAN into this repository—contributions are always welcome!

# Kolmogorov-Arnold Networks
Kolmogorov-Arnold Networks (KANs) are a type of neural network architecture inspired by the Kolmogorov-Arnold Representation Theorem. This theorem states that any continuous multivariable function can be represented as a superposition of a finite number of univariate functions.
## Papers
- **KAN: Kolmogorov-Arnold Networks**: https://arxiv.org/abs/2404.19756
- **KAN 2.0: Kolmogorov-Arnold Networks Meet Science**: https://arxiv.org/abs/2408.10205

# Existing KANs
## My KANs
These are KANs that I have developed:
- BSRBF-KAN: https://github.com/hoangthangta/BSRBF_KAN
- FC-KAN: https://github.com/hoangthangta/FC_KAN

## Other KANs
You can open a pull to add your KANs in this section.
- PyKAN (Original KAN, Spl-KAN, LiuKAN): https://github.com/KindXiaoming/pykan
- FastKAN: https://github.com/ZiyaoLi/fast-kan
- FasterKAN: https://github.com/AthanasiosDelis/faster-kan
- Wav-KAN: https://github.com/zavareh1/Wav-KAN
- More KANs: https://github.com/mintisan/awesome-kan

# Experiments
## Requirements 
* numpy==1.26.4
* numpyencoder==0.3.0
* torch==2.3.0+cu118
* torchvision==0.18.0+cu118
* tqdm==4.66.4

## Parameters
* *mode*: working mode ("train").
* *ds_name*: dataset name ("mnist" or "fashion_mnist").
* *model_name*: type of models (*bsrbf_kan*, *efficient_kan*, *fast_kan*, *faster_kan*, *mlp*, and *fc_kan*).
* *epochs*: the number of epochs.
* *batch_size*: the training batch size (default: 64).
* *n_input*: The number of input neurons (default: 28^2 = 784).
* *n_hidden*: The number of hidden neurons. We use only 1 hidden layer. You can modify the code (**run.py**) for more layers.
* *n_output*: The number of output neurons (classes). For MNIST and Fashion-MNIST, there are 10 classes.
* *grid_size*: The size of the grid (default: 5). Use with bsrbf_kan and efficient_kan.
* *spline_order*: The order of spline (default: 3). Use with bsrbf_kan and efficient_kan.
* *num_grids*: The number of grids, equals grid_size + spline_order (default: 8). Use with fast_kan and faster_kan.
* *device*: use "cuda" or "cpu" (default: "cuda").
* *n_examples*: the number of examples in the training set used for training (default: 0, mean use all training data)
* *note*: A note saved in the model name file.
* *n_part*: the part of data used to train data (default: 0, mean use all training data, 0.1 means 10%).
* *func_list*: the name of functions used in FC-KAN (default='dog,rbf'). Other functions are *bs* and *base*, and functions in SKAN (*shifted_softplus*, *arctan*, *relu*, *elu*, *gelup*, *leaky_relu*, *swish*, *softplus*, *sigmoid*, *hard_sigmoid*, *sinv, *cos*). 
* *combined_type*: the type of data combination used in the output (default='quadratic', others are *sum*, *product*, *sum_product*, *concat*, *max*, *min*, *mean*).

## Commands
### BSRBF-KAN, FastKAN, FasterKAN, GottliebKAN, and MLP
```python run.py --mode "train" --ds_name "mnist" --model_name "bsrbf_kan" --epochs 25 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --grid_size 5 --spline_order 3```

```python run.py --mode "train" --ds_name "mnist" --model_name "efficient_kan" --epochs 25 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --grid_size 5 --spline_order 3```

```python run.py --mode "train" --ds_name "mnist" --model_name "fast_kan" --epochs 25 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --num_grids 8```

```python run.py --mode "train" --ds_name "mnist" --model_name "faster_kan" --epochs 25 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --num_grids 8```

```python run.py --mode "train" --ds_name "mnist" --model_name "gottlieb_kan" --epochs 25 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --spline_order 3```

```python run.py --mode "train" --ds_name "mnist" --model_name "mlp" --epochs 25 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10```

### FC-KAN
FC-KAN models (Difference of Gaussians + B-splines) can be trained on MNIST with different output combinations as follows.

```python run.py --mode "train" --model_name "fc_kan" --epochs 25 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --ds_name "mnist" --note "full_0" --n_part 0 --func_list "dog,bs" --combined_type "sum"```

```python run.py --mode "train" --model_name "fc_kan" --epochs 25 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --ds_name "mnist" --note "full_0" --n_part 0 --func_list "dog,bs" --combined_type "product"```

```python run.py --mode "train" --model_name "fc_kan" --epochs 25 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --ds_name "mnist" --note "full_0" --n_part 0 --func_list "dog,bs" --combined_type "sum_product"```

```python run.py --mode "train" --model_name "fc_kan" --epochs 25 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --ds_name "mnist" --note "full_0" --n_part 0 --func_list "dog,bs" --combined_type "quadratic"```

```python run.py --mode "train" --model_name "fc_kan" --epochs 25 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --ds_name "mnist" --note "full_0" --n_part 0 --func_list "dog,bs" --combined_type "concat"```

### PRKAN
Updating...
