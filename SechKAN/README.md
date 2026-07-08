## Running SechKAN

This directory contains the implementation of **SechKAN: Kolmogorov–Arnold Networks with Hyperbolic Secant Functions**.

> **Important:** All commands should be executed from the `main` directory, **not** from this directory.

SechKAN supports the following tasks:

- **Function fitting** – Run `run_ff.py`.
- **Image classification** (MNIST, Fashion-MNIST, CIFAR-10, and CIFAR-100) – Run `run.py`. See `run_ic_sech_kan.sh` for the commands used to reproduce the image classification experiments with SechKAN.
- **PDE surrogate modeling** (Navier–Stokes and Shallow Water) – Run `run_navier.py` and `run_shallow_water.py`.
