![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-2.3.0%2B-red?logo=pytorch&logoColor=white) ![Badge](https://img.shields.io/badge/Newest-RationalKAN-<blue>)

<p align="center">
<img src="https://github.com/hoangthangta/All-KAN/blob/main/All_KAN.png" width="400" />
</p>

# Plagiarism behaviors
We strongly condemn the plagiarism committed by the authors of the work titled **"CPR-KANsformer: An Adaptive Multi-dimensional CSI Feature Extraction Model for Human Activity Recognition" ([link](https://link.springer.com/chapter/10.1007/978-981-96-9955-1_38))**. They have copied substantial elements from our original work, "PRKAN: Parameter-Reduced Kolmogorov-Arnold Networks" ([arXiv link](https://arxiv.org/abs/2501.07032)), including the network name "PRKAN", methodological details, and even figures.

# News
- **8-9 Mar, 2025**: Created AF-KAN, and updated ReLU-KAN, ChebyKAN, FourierKAN, KnotsKAN, RationalKAN, RBF_KAN.
- **23 Mar, 2025**: Update function fitting (run_ff.py) and code.

# To Do List
- https://github.com/Adamdad/kat
- https://github.com/lif314/X-KANeRF
- https://github.com/Jinfeng-Xu/FKAN-GCF

# All-KAN: All Kolmogorov-Arnold Networks We Know
This repository gathers all known Kolmogorov-Arnold Networks (including those I developed) from various sources. These networks are implemented for **image classification on some simple image classification datasets (MNIST, Fashion-MNIST, CIFAR10, etc)**. I hope this collection inspires you and the broader research community to advance the development of even better KANs in the future.

Feel free to suggest or integrate your KAN into this repository—contributions are always welcome!

# Kolmogorov-Arnold Networks
Kolmogorov-Arnold Networks (KANs) are a type of neural network architecture inspired by the Kolmogorov-Arnold Representation Theorem. This theorem states that any continuous multivariable function can be represented as a superposition of a finite number of univariate functions.

## Papers
- **KAN: Kolmogorov-Arnold Networks**: https://arxiv.org/abs/2404.19756
- **KAN 2.0: Kolmogorov-Arnold Networks Meet Science**: https://arxiv.org/abs/2408.10205
- **PRKAN: Parameter-Reduced Kolmogorov-Arnold Networks**: https://arxiv.org/abs/2501.07032
- **BSRBF-KAN: A combination of B-splines and Radial Basis Functions in Kolmogorov-Arnold Networks**: https://arxiv.org/abs/2406.11173
- **FC-KAN: Function Combinations in Kolmogorov-Arnold Networks**: https://arxiv.org/abs/2409.01763
- **AF-KAN: Activation Function-Based Kolmogorov-Arnold Networks for Efficient Representation Learning**: https://arxiv.org/abs/2503.06112
  
## Advantages of KANs
- Perform exceptionally well on tasks involving mathematics, physics, and complex curves, such as function fitting and symbolic regression, often outperforming traditional MLPs. So, consider using KANs with data that contains curves.
- Capture richer and more meaningful data features, leading to more expressive representations and improved overall model performance. 
- Inherent interpretability, as KANs use spline-based or polynomial-based functions, making their internal representations more understandable compared to black-box MLPs.
- Better generalization for smooth functions, due to their ability to approximate functions with fewer layers while preserving accuracy.
## Problem of KANs
- *Parameter inefficiency*: KANs typically require more parameters than MLPs because they capture data features using polynomial functions such as B-splines.
- *Longer training time*: Due to parameter inefficiency, the training time is longer.
- *Less Scalability*: Again, due to parameter inefficiency, adding more layers to KANs results in bulky networks. This is why some suggest that KANs should be designed with fewer layers.
## Proper use of KANs in modern neural networks
While KANs can function as standalone networks for certain problems, they are often used as components within neural architectures such as CNNs, RNNs, or Transformers, where they can replace MLP layers, convolutional layers, or other components. In this role, KANs act as a supplement to enhance the overall performance of the network.
## Why we must use KANs?
Simply put, KANs offer a different approach to capturing data compared to MLPs. Instead of relying on linear mappings as in MLPs, KANs use diverse polynomial functions and their combinations to represent data. **This encourages researchers and practitioners to explore more the role of mathematical functions in designing neural networks, which is a positive trend that could lead to the development of more diverse neural network architectures in the future.**

If you are bored with MLPs, which have been developed over several decades, KANs offer a fresh and innovative alternative to explore.

# Existing KANs
## My KANs
These are KANs that I have developed:
- BSRBF-KAN: Code is in this repo, other place is: https://github.com/hoangthangta/BSRBF_KAN.
- FC-KAN: Code is in this repo, other place is: https://github.com/hoangthangta/FC_KAN.
- PRKAN: Code is in this repo.
- AF-KAN: Code is in this repo.

## Other KANs
You can open a pull to add your KANs in this section.
- PyKAN (Original KAN, Spl-KAN, LiuKAN): https://github.com/KindXiaoming/pykan
- FastKAN: https://github.com/ZiyaoLi/fast-kan
- FasterKAN: https://github.com/AthanasiosDelis/faster-kan
- Wav-KAN: https://github.com/zavareh1/Wav-KAN
- ReLUKAN: https://github.com/quiqi/relu_kan
- SKAN: https://github.com/chikkkit/LSS-SKAN, https://github.com/chikkkit/LArctan-SKAN
- ChebyKAN, FourierKAN, KnotsKAN, RationalKAN, RBF_KAN: https://github.com/icuraslw/fr-kan
- More KANs: https://github.com/mintisan/awesome-kan

# Experiments
## Requirements 
* numpy==1.26.4
* numpyencoder==0.3.0
* torch==2.3.0+cu118
* torchvision==0.18.0+cu118
* tqdm==4.66.4

## Parameters
### Image Classification (run.py)
* *mode*: working mode ("train" or "predict_set").
* *ds_name*: dataset name ("mnist", "fashion_mnist", "cifar10"). Note: We did not test the performance on CIFAR10.
* *model_name*: type of models (*bsrbf_kan*, *efficient_kan*, *fast_kan*, *faster_kan*, *mlp*, and *fc_kan*, etc.).
* *epochs*: the number of epochs.
* *batch_size*: the training batch size (default: 64).
* *layers*: network layers (default: "784,64,10" = 784 input nodes, 64 hidden nodes, and 10 output nodes).
* *lr*: learning rate (default: 1e-3).
* *wc*: weight decay (default: 1e-4).
* *scheduler*: learning rate scheduler (default: 'ExponentialLR', others are *StepLR*, *CosineAnnealingLR*, **OneCycleLR**, *CyclicLR*).
* *grid_size*: The size of the grid (default: 5). Favor using bsrbf_kan, efficient_kan, and other variants that leverage B-splines or similar functions.
* *spline_order*: The order of spline (default: 3). Favor using bsrbf_kan, efficient_kan and other KAN variants that leverage B-splines or similar functions.
* *num_grids*: The number of grids, equals grid_size + spline_order (default: 8). Favor using fast_kan and faster_kan models based on Radial Basis Functions (RBFs).
* *device*: use "cuda" or "cpu" (default: "cuda").
* *n_examples*: the number of examples in the training set used for training (default: 0, mean use all training data)
* *note*: A note saved in the model name file.
* *n_part*: the part of data used to train data (default: 0, mean use all training data, 0.1 means 10%).
* *func_list*: the name of functions used in FC-KAN (default='dog,rbf'). Other functions are *bs* and *base*, and functions in SKAN (*shifted_softplus*, *arctan*, *relu*, *elu*, *gelup*, *leaky_relu*, *swish*, *softplus*, *sigmoid*, *hard_sigmoid*, *cos*). 
* *combined_type*: the type of data combination used in the output (default='quadratic', others are *sum*, *product*, *sum_product*, *concat*, *max*, *min*, *mean*).
* *basis_function*: used in SKAN (default='sin', others are *shifted_softplus*, *arctan*, *relu*, *elu*, *gelup*, *leaky_relu*, *swish*, *softplus*, *sigmoid*, *hard_sigmoid*, *cos*).
* *func*: the basis function, used in PRKAN (default='rbf', other is *layer*)
* *methods*: reduction methods, used in PRKAN (default='attention', other are *conv1d_1* = convolution, *conv1d_2* = convolution + pooling, *attention*, *fw* = feature weight vector, *ds* = dim-sum) or AF-KAN (*global_attn*, *spatial_attn*, *multistep*, and more, check its code.)
* *norm_type*:  data normalization type, used in PRKAN, AF-KAN, ReLU-KAN (default=*layer*, other is *batch*, *none*)
* *base_activation*: base activation, used in PRKAN, AF-KAN, ReLU-KAN (default='silu', other are *selu*, *gelu*, *elu*, *silu*, *relu*, *softplus*, *sigmoid*, *leaky_relu*)
* *norm_pos*: data normalization position, used in PRKAN (default=1, other is *2*)
* *func*: function types used in AF-KAN (*quad1*, *quad2*, *sum*, *prod*, *sum_prod*, *cubic1*, *cubic2*)
* *p_order*: the order of P function, used in RationalKAN (default=3)
* *q_order*:  the order of Q function, used in RationalKAN (default=3)
* *groups*: number of groups used in RationalKAN (default=8)
### Function fitting (run_ff.py)
* *mode*: working mode ("train").
* *model_name*: type of models (*bsrbf_kan*, *efficient_kan*, *fast_kan*, *faster_kan*, *mlp*, and *fc_kan*, etc.).
* *layers*: network layers.
* *func*: function types used for fitting, check run_ff.py to see all functions.
* *epochs*: the number of epochs.

## Commands
### BSRBF-KAN, FastKAN, FasterKAN, GottliebKAN, and MLP
For BSBRF-KAN, also see: https://github.com/hoangthangta/BSRBF_KAN.
<details>
<summary><b>Click here for code!</b></summary>

```python run.py --mode "train" --ds_name "mnist" --model_name "bsrbf_kan" --epochs 25 --batch_size 64 --layers "784,64,10" --grid_size 5 --spline_order 3```

```python run.py --mode "train" --ds_name "mnist" --model_name "efficient_kan" --epochs 25 --batch_size 64 --layers "784,64,10" --grid_size 5 --spline_order 3```

```python run.py --mode "train" --ds_name "mnist" --model_name "fast_kan" --epochs 25 --batch_size 64 --layers "784,64,10" --num_grids 8```

```python run.py --mode "train" --ds_name "mnist" --model_name "faster_kan" --epochs 25 --batch_size 64 --layers "784,64,10" --num_grids 8```

```python run.py --mode "train" --ds_name "mnist" --model_name "gottlieb_kan" --epochs 25 --batch_size 64 --layers "784,64,10" --spline_order 3```

```python run.py --mode "train" --ds_name "mnist" --model_name "mlp" --epochs 25 --batch_size 64 --layers "784,64,10"```

</details>


### FC-KAN
For FC-KAN, also see: https://github.com/hoangthangta/FC_KAN. FC-KAN models (Difference of Gaussians + B-splines) can be trained on MNIST with different output combinations as follows.

<details>
<summary><b>Click here for code!</b></summary>

```python run.py --mode "train" --model_name "fc_kan" --epochs 25 --batch_size 64 --layers "784,64,10" --ds_name "mnist" --note "full_0" --n_part 0 --func_list "dog,bs" --combined_type "sum"```

```python run.py --mode "train" --model_name "fc_kan" --epochs 25 --batch_size 64 --layers "784,64,10" --ds_name "mnist" --note "full_0" --n_part 0 --func_list "dog,bs" --combined_type "product"```

```python run.py --mode "train" --model_name "fc_kan" --epochs 25 --batch_size 64 --layers "784,64,10" --ds_name "mnist" --note "full_0" --n_part 0 --func_list "dog,bs" --combined_type "sum_product"```

```python run.py --mode "train" --model_name "fc_kan" --epochs 25 --batch_size 64 --layers "784,64,10" --ds_name "mnist" --note "full_0" --n_part 0 --func_list "dog,bs" --combined_type "quadratic"```

```python run.py --mode "train" --model_name "fc_kan" --epochs 25 --batch_size 64 --layers "784,64,10" --ds_name "mnist" --note "full_0" --n_part 0 --func_list "dog,bs" --combined_type "concat"```
</details>

### SKAN
<details>
<summary><b>Click here for code!</b></summary>
  
```python run.py --mode "train" --model_name "skan" --epochs 25 --batch_size 64 --layers "784,64,10" --ds_name "mnist" --basis_function "arctan"```

```python run.py --mode "train" --model_name "skan" --epochs 35 --batch_size 64 --layers "784,64,10" --ds_name "fashion_mnist" --basis_function "arctan"```
</details>

### PRKAN
<details>
<summary><b>Click here for code!</b></summary>

```python run.py --mode "train" --model_name "prkan" --epochs 25 --batch_size 64 --layers "784,64,10" --ds_name "mnist" --note "full_0" --n_part 0 --func "rbf" --base_activation "silu" --methods "conv1d_1" --norm_type "layer" --norm_pos 1;```

```python run.py --mode "train" --model_name "prkan" --epochs 25 --batch_size 64 --layers "784,64,10" --ds_name "mnist" --note "full_0" --n_part 0 --func "rbf" --base_activation "silu" --methods "conv1d_2" --norm_type "layer" --norm_pos 1;```

```python run.py --mode "train" --model_name "prkan" --epochs 25 --batch_size 64 --layers "784,64,10" --ds_name "mnist" --note "full_0" --n_part 0 --func "rbf" --base_activation "silu" --methods "attention" --norm_type "layer" --norm_pos 2;```

</details>

### ReLUKAN
ReLUKAN is better with **grid_size=3** and **spline_order=3**.
<details>
<summary><b>Click here for code!</b></summary>

```python run.py --mode "train" --model_name "relu_kan" --epochs 25 --batch_size 64 --layers "784,64,10" --grid_size 3 --spline_order 3 --ds_name "mnist" --norm_type "layer";```

```python run.py --mode "train" --model_name "relu_kan" --epochs 25 --batch_size 64 --layers "784,64,10" --grid_size 3 --spline_order 3 --ds_name "mnist" --norm_type "batch";```

```python run.py --mode "train" --model_name "relu_kan" --epochs 35 --batch_size 64 --layers "784,64,10" --grid_size 3 --spline_order 3 --ds_name "fashion_mnist" --norm_type "layer";```

```python run.py --mode "train" --model_name "relu_kan" --epochs 35 --batch_size 64 --layers "784,64,10" --grid_size 3 --spline_order 3 --ds_name "fashion_mnist" --norm_type "batch";```

</details>

### AF-KAN
AF-KAN is better with **grid_size=3** and **spline_order=3**.
<details>
<summary><b>Click here for code!</b></summary>

```python run.py --mode "train" --model_name "af_kan" --epochs 25 --batch_size 64 --layers "784,64,10" --grid_size 3 --spline_order 3 --ds_name "mnist" --note "full" --n_part 0 --base_activation "silu" --norm_type "layer" --method "global_attn" --func "quad1";```

```python run.py --mode "train" --model_name "af_kan" --epochs 35 --batch_size 64 --layers "784,64,10" --grid_size 3 --spline_order 3 --ds_name "fashion_mnist" --note "full" --n_part 0 --base_activation "silu" --norm_type "layer" --method "global_attn" --func "quad1";```

</details>

### ChebyKAN
Fail with Fashion-MNIST? 
<details>
<summary><b>Click here for code!</b></summary>

  ```python run.py --mode "train" --model_name "cheby_kan" --epochs 25 --batch_size 64 --layers "784,64,10" --spline_order 3 --ds_name "mnist"```
  
```python run.py --mode "train" --model_name "cheby_kan" --epochs 35 --batch_size 64 --layers "784,64,10" --spline_order 3 --ds_name "fashion_mnist"```
</details>

### FourierKAN
Fail with MNIST + Fashion-MNIST?
<details>
<summary><b>Click here for code!</b></summary>

```python run.py --mode "train" --model_name "fourier_kan" --epochs 25 --batch_size 64 --layers "784,64,10" --grid_size 5 --spline_order 3 --ds_name "mnist"```
  
```python run.py --mode "train" --model_name "fourier_kan" --epochs 35 --batch_size 64 --layers "784,64,10" --grid_size 5 --spline_order 3 --ds_name "fashion_mnist"```
</details>

### KnotsKAN
Default **grid_size=20**.
<details>
<summary><b>Click here for code!</b></summary>

```python run.py --mode "train" --model_name "knots_kan" --epochs 25 --batch_size 64 --layers "784,64,10" --grid_size 5 --spline_order 3 --ds_name "mnist"```
  
```python run.py --mode "train" --model_name "knots_kan" --epochs 35 --batch_size 64 --layers "784,64,10" --grid_size 5 --spline_order 3 --ds_name "fashion_mnist"```
</details>

### RationalKAN
<details>
<summary><b>Click here for code!</b></summary>

```python run.py --mode "train" --model_name "rational_kan" --epochs 25 --batch_size 64 --layers "784,64,10" --ds_name "mnist" --p_order 3 --q_order 3 --groups 8```
  
```python run.py --mode "train" --model_name "rational_kan" --epochs 35 --batch_size 64 --layers "784,64,10" --ds_name "fashion_mnist" --p_order 3 --q_order 3 --groups 8```
</details>

### RBF_KAN
Similar to FastKAN
<details>
<summary><b>Click here for code!</b></summary>

```python run.py --mode "train" --model_name "rbf_kan" --epochs 25 --batch_size 64 --layers "784,64,10" --ds_name "mnist" --grid_size 5 --spline_order 3```
  
```python run.py --mode "train" --model_name "rbf_kan" --epochs 35 --batch_size 64 --layers "784,64,10" --ds_name "fashion_mnist" --grid_size 5 --spline_order 3```
</details>


# Acknowledgement
Thank you all for your contributions and suggestions to this repo. If you like it, please consider giving us a star. Thanks! <3 <3 <3

# How to cite?

You can cite these papers.

```
@misc{ta2025afkanactivationfunctionbasedkolmogorovarnold,
      title={AF-KAN: Activation Function-Based Kolmogorov-Arnold Networks for Efficient Representation Learning}, 
      author={Hoang-Thang Ta and Anh Tran},
      year={2025},
      eprint={2503.06112},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.06112}, 
}
```

```
@article{ta2025prkan,
  title={PRKAN: Parameter-Reduced Kolmogorov-Arnold Networks},
  author={Ta, Hoang-Thang and Thai, Duy-Quy and Tran, Anh and Sidorov, Grigori and Gelbukh, Alexander},
  journal={arXiv preprint arXiv:2501.07032},
  year={2025}
}
```

```
@article{ta2024fc,
  title={FC-KAN: Function Combinations in Kolmogorov-Arnold Networks},
  author={Ta, Hoang-Thang and Thai, Duy-Quy and Rahman, Abu Bakar Siddiqur and Sidorov, Grigori and Gelbukh, Alexander},
  journal={arXiv preprint arXiv:2409.01763},
  year={2024}
}
```

```
@article{ta2024bsrbf,
  title={BSRBF-KAN: A combination of B-splines and Radial Basis Functions in Kolmogorov-Arnold Networks},
  author={Ta, Hoang-Thang},
  journal={arXiv preprint arXiv:2406.11173},
  year={2024}
}
```
