import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import random
from tqdm import tqdm
import copy
import time
from utils import *
from models import EfficientKAN, FastKAN, BSRBF_KAN, FasterKAN, ReLUKAN, SechKAN, MLP
import os
from datetime import datetime
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model, loader, criterion, y_mean=None, y_std=None):
    model.eval()

    total_loss = 0.0
    sq_error_sum = 0.0
    rel_num = 0.0
    rel_den = 0.0
    n_samples = 0

    eps = 1e-8
    out_dim = None

    if y_mean is not None and y_std is not None:
        y_mean_t = torch.as_tensor(y_mean, device=device, dtype=torch.float32)
        y_std_t  = torch.as_tensor(y_std, device=device, dtype=torch.float32)
    else:
        y_mean_t = None
        y_std_t = None

    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            yb = yb.to(device)

            pred = model(Xb)

            loss = criterion(pred, yb)
            total_loss += loss.item() * Xb.size(0)

            if y_mean_t is not None:
                pred = pred * y_std_t + y_mean_t
                yb   = yb * y_std_t + y_mean_t

            diff = pred - yb

            if out_dim is None:
                out_dim = yb.shape[1]

            err_sq = torch.sum(diff ** 2).item()
            sq_error_sum += err_sq
            rel_num += err_sq

            rel_den += torch.sum(yb ** 2).item()

            n_samples += Xb.size(0)

    rmse = np.sqrt(sq_error_sum / (n_samples * out_dim))
    rel_l2 = np.sqrt(rel_num / (rel_den + eps))
    avg_loss = total_loss / n_samples

    return rmse, rel_l2, avg_loss


def generate_navier_stokes(n_x=64, n_y=64, n_t=60,
                           nu=1e-3,
                           dt=1e-2,
                           forcing=True,
                           seed=42):

    np.random.seed(seed)

    # domain
    Lx = Ly = 2 * np.pi
    x = np.linspace(0, Lx, n_x, endpoint=False)
    y = np.linspace(0, Ly, n_y, endpoint=False)
    t = np.arange(n_t) * dt

    X, Y = np.meshgrid(x, y, indexing='ij')

    # wave numbers (physical)
    kx = 2*np.pi * np.fft.fftfreq(n_x, d=Lx/n_x)
    ky = 2*np.pi * np.fft.fftfreq(n_y, d=Ly/n_y)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')

    K2 = KX**2 + KY**2
    #inv_K2 = np.where(K2 == 0, 0.0, 1.0 / K2)
    inv_K2 = np.zeros_like(K2)
    inv_K2[K2 != 0] = 1.0 / K2[K2 != 0]

    # dealiasing mask (2/3 rule)
    kx_idx = np.fft.fftfreq(n_x) * n_x
    ky_idx = np.fft.fftfreq(n_y) * n_y
    KX_idx, KY_idx = np.meshgrid(kx_idx, ky_idx, indexing='ij')

    dealias_mask = (
        (np.abs(KX_idx) < (2/3) * (n_x // 2)) &
        (np.abs(KY_idx) < (2/3) * (n_y // 2))
    ).astype(float)

    # initial vorticity (spectrally filtered)
    omega = np.random.randn(n_x, n_y)
    omega_hat = np.fft.fft2(omega) * dealias_mask

    U = np.zeros((n_t, n_x, n_y))
    V = np.zeros((n_t, n_x, n_y))

    # velocity from streamfunction
    def compute_velocity(w_hat):
        psi_hat = -w_hat * inv_K2
        u_hat = 1j * KY * psi_hat
        v_hat = -1j * KX * psi_hat
        return np.fft.ifft2(u_hat).real, np.fft.ifft2(v_hat).real

    # nonlinear term
    def nonlinear(w_hat):
        u, v = compute_velocity(w_hat)

        w = np.fft.ifft2(w_hat).real
        wx = np.fft.ifft2(1j * KX * w_hat).real
        wy = np.fft.ifft2(1j * KY * w_hat).real

        adv = u * wx + v * wy
        return np.fft.fft2(-adv) * dealias_mask

    # forcing
    def forcing_term(tt):
        if not forcing:
            return 0.0
        f = np.sin(X + 0.5 * tt) * np.cos(Y)
        return np.fft.fft2(f) * dealias_mask

    # RHS
    def rhs(w_hat, tt):
        return (
            nonlinear(w_hat)
            + nu * (-K2) * w_hat
            + forcing_term(tt)
        )

    # RK4 step
    def rk4(w_hat, tt):

        k1 = rhs(w_hat, tt)
        k2 = rhs(w_hat + 0.5*dt*k1, tt + 0.5*dt)
        k3 = rhs(w_hat + 0.5*dt*k2, tt + 0.5*dt)
        k4 = rhs(w_hat + dt*k3, tt + dt)

        return w_hat + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

    # time integration
    for i in range(n_t):

        u, v = compute_velocity(omega_hat)
        U[i] = u
        V[i] = v

        omega_hat = rk4(omega_hat, t[i])

    return X, Y, t, U, V


def prepare_data(X, Y, T, U, V, batch_size=256, seed=42, split_mode="shuffle"):

    np.random.seed(seed)
    random.seed(seed)

    n_t, n_x, n_y = U.shape

    n_train_t = int(0.7 * n_t)
    n_val_t   = int(0.15 * n_t)

    idx = np.arange(n_t)

    if split_mode == "shuffle":
        idx = np.random.permutation(idx)

    train_idx = idx[:n_train_t]
    val_idx   = idx[n_train_t:n_train_t + n_val_t]
    test_idx  = idx[n_train_t + n_val_t:]

    def build_split(t_idx):
        xx, yy = X, Y

        inputs_list = []
        targets_list = []

        for t in t_idx:
            tt = np.full_like(xx, T[t])

            inputs_list.append(np.stack([xx, yy, tt], axis=-1).reshape(-1, 3))
            targets_list.append(np.stack([U[t], V[t]], axis=-1).reshape(-1, 2))

        inputs = np.concatenate(inputs_list, axis=0)
        targets = np.concatenate(targets_list, axis=0)

        return inputs, targets
    

    train_inputs, train_targets = build_split(train_idx)
    val_inputs, val_targets     = build_split(val_idx)
    test_inputs, test_targets   = build_split(test_idx)

    # Convert to tensors
    train_x = torch.from_numpy(train_inputs).float()
    train_y = torch.from_numpy(train_targets).float()

    val_x = torch.from_numpy(val_inputs).float()
    val_y = torch.from_numpy(val_targets).float()

    test_x = torch.from_numpy(test_inputs).float()
    test_y = torch.from_numpy(test_targets).float()

    # Normalization (Train only)
    x_mean = train_x.mean(dim=0, keepdim=True)
    x_std  = train_x.std(dim=0, keepdim=True) + 1e-6

    y_mean = train_y.mean(dim=0, keepdim=True)
    y_std  = train_y.std(dim=0, keepdim=True) + 1e-6

    train_x = (train_x - x_mean) / x_std
    val_x   = (val_x - x_mean) / x_std
    test_x  = (test_x - x_mean) / x_std

    train_y = (train_y - y_mean) / y_std
    val_y   = (val_y - y_mean) / y_std
    test_y  = (test_y - y_mean) / y_std

    # Dataloaders
    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        TensorDataset(val_x, val_y),
        batch_size=batch_size * 2,
        shuffle=False,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        TensorDataset(test_x, test_y),
        batch_size=batch_size * 2,
        shuffle=False,
        pin_memory=torch.cuda.is_available()
    )

    # Print results
    print("=" * 50)
    print("DATASET")
    print("=" * 50)
    print(f"Split mode: {split_mode}")
    print(f"Train:      {len(train_loader.dataset):,}")
    print(f"Val:        {len(val_loader.dataset):,}")
    print(f"Test:       {len(test_loader.dataset):,}")
    print("Ratio:       70 / 15 / 15")
    print("=" * 50)

    return train_loader, val_loader, test_loader, y_mean.numpy(), y_std.numpy()


def build_models(in_dim, out_dim):
    return {
        "SechKAN4": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="",
            norm2_type="",
            base_activation="silu",
            use_width=False,
            norm_mode="except_first",
            num_grids=4,
            use_base_update=False
        ).to(device),
        "SechKAN5": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="layer",
            norm2_type="",
            base_activation="silu",
            use_width=False,
            norm_mode="except_first",
            num_grids=4,
            use_base_update=False
        ).to(device),
        "SechKAN6": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="",
            norm2_type="mm",
            base_activation="silu",
            use_width=False,
            norm_mode="except_first",
            num_grids=4,
            use_base_update=False
        ).to(device),
    }
    

'''def build_models(in_dim, out_dim):
    return {
        "SechKAN1": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="",
            norm2_type="",
            base_activation="silu",
            use_width=False,
            norm_mode="except_first",
            num_grids=8,
            use_base_update=False
        ).to(device),
        "SechKAN2": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="layer",
            norm2_type="",
            base_activation="silu",
            use_width=False,
            norm_mode="except_first",
            num_grids=8,
            use_base_update=False
        ).to(device),
        "SechKAN3": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="",
            norm2_type="mm",
            base_activation="silu",
            use_width=False,
            norm_mode="except_first",
            num_grids=8,
            use_base_update=False
        ).to(device),
        "EfficientKAN": lambda: EfficientKAN([in_dim, 61, 60, out_dim],
                                              grid_size=5, spline_order=3).to(device),
        "FastKAN": lambda: FastKAN([in_dim, 63, 63, out_dim], num_grids=8, use_layernorm = "").to(device),
        "MLP": lambda: MLP([in_dim, 196, 196, out_dim], base_activation="silu", norm_type = "").to(device),
        "ReLUKAN": lambda: ReLUKAN([in_dim, 66, 66, out_dim], grid=5, k=3, norm_type = "").to(device),
        "BSRBF_KAN": lambda: BSRBF_KAN([in_dim, 64, 64, out_dim],
                                        grid_size=5, spline_order=3, norm_type = "").to(device),
        "FasterKAN": lambda: FasterKAN([in_dim, 68, 67, out_dim], num_grids=8, norm_type = "").to(device)
    }
'''
    
    
# Ablation: num_grids (seed 0 & 1)
'''def build_models(in_dim, out_dim):
    return {
        
        "SechKAN5_num_grids_4": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="layer",
            norm2_type="",
            base_activation="silu",
            use_width=False,
            norm_mode="except_first",
            num_grids=4,
            use_base_update=False
        ).to(device),
        "SechKAN5_num_grids_2": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="layer",
            norm2_type="",
            base_activation="silu",
            use_width=False,
            norm_mode="except_first",
            num_grids=2,
            use_base_update=False
        ).to(device),
        "SechKAN5_num_grids_8": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="layer",
            norm2_type="",
            base_activation="silu",
            use_width=False,
            norm_mode="except_first",
            num_grids=8,
            use_base_update=False
        ).to(device),
        "SechKAN5_num_grids_16": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="layer",
            norm2_type="",
            base_activation="silu",
            use_width=False,
            norm_mode="except_first",
            num_grids=16,
            use_base_update=False
        ).to(device),
        "SechKAN5_num_grids_32": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="layer",
            norm2_type="",
            base_activation="silu",
            use_width=False,
            norm_mode="except_first",
            num_grids=32,
            use_base_update=False
        ).to(device),
        
    }'''


# Ablation: Activations (seed 0 & 1)
'''def build_models(in_dim, out_dim):
    return {
        
        "SechKAN5_norm1_layer_silu": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="layer",
            norm2_type="",
            base_activation="silu",
            use_width=False,
            norm_mode="except_first",
            num_grids=4,
            use_base_update=False
        ).to(device),
        "SechKAN5_norm1_layer_softplus": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="layer",
            norm2_type="",
            base_activation="softplus",
            use_width=False,
            norm_mode="except_first",
            num_grids=4,
            use_base_update=False
        ).to(device),
        "SechKAN5_norm1_layer_sigmoid": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="layer",
            norm2_type="",
            base_activation="sigmoid",
            use_width=False,
            norm_mode="except_first",
            num_grids=4,
            use_base_update=False
        ).to(device),
        "SechKAN5_norm1_layer_relu": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="layer",
            norm2_type="",
            base_activation="relu",
            use_width=False,
            norm_mode="except_first",
            num_grids=4,
            use_base_update=False
        ).to(device),
        "SechKAN5_norm1_layer_leaky_relu": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="layer",
            norm2_type="",
            base_activation="leaky_relu",
            use_width=False,
            norm_mode="except_first",
            num_grids=4,
            use_base_update=False
        ).to(device),
        "SechKAN5_norm1_layer_elu": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="layer",
            norm2_type="",
            base_activation="elu",
            use_width=False,
            norm_mode="except_first",
            num_grids=4,
            use_base_update=False
        ).to(device),
        "SechKAN5_norm1_layer_gelu": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="layer",
            norm2_type="",
            base_activation="gelu",
            use_width=False,
            norm_mode="except_first",
            num_grids=4,
            use_base_update=False
        ).to(device),
        "SechKAN5_norm1_layer_selu": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="layer",
            norm2_type="",
            base_activation="selu",
            use_width=False,
            norm_mode="except_first",
            num_grids=4,
            use_base_update=False
        ).to(device),
        "SechKAN5_norm1_layer_tanh": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="layer",
            norm2_type="",
            base_activation="tanh",
            use_width=False,
            norm_mode="except_first",
            num_grids=4,
            use_base_update=False
        ).to(device),

    }
'''

# Ablation: norm (seed 0 & 1)
'''def build_models(in_dim, out_dim):
    return {
        
        "SechKAN5_norm1_layer": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="layer",
            norm2_type="",
            base_activation="silu",
            use_width=False,
            norm_mode="except_first",
            num_grids=4,
            use_base_update=False
        ).to(device),
        "SechKAN5_norm1_batch": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="batch",
            norm2_type="",
            base_activation="silu",
            use_width=False,
            norm_mode="except_first",
            num_grids=4,
            use_base_update=False
        ).to(device),
        "SechKAN5_norm1_mm": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="mm",
            norm2_type="",
            base_activation="silu",
            use_width=False,
            norm_mode="except_first",
            num_grids=4,
            use_base_update=False
        ).to(device),
        "SechKAN5_norm1_rms": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="rms",
            norm2_type="",
            base_activation="silu",
            use_width=False,
            norm_mode="except_first",
            num_grids=4,
            use_base_update=False
        ).to(device),
        "SechKAN5_none": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="",
            norm2_type="",
            base_activation="silu",
            use_width=False,
            norm_mode="except_first",
            num_grids=4,
            use_base_update=False
        ).to(device),
        "SechKAN5_norm2_layer": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="",
            norm2_type="layer",
            base_activation="silu",
            use_width=False,
            norm_mode="except_first",
            num_grids=4,
            use_base_update=False
        ).to(device),
        "SechKAN5_norm2_batch": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="",
            norm2_type="batch",
            base_activation="silu",
            use_width=False,
            norm_mode="except_first",
            num_grids=4,
            use_base_update=False
        ).to(device),
        "SechKAN5_norm2_mm": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="",
            norm2_type="mm",
            base_activation="silu",
            use_width=False,
            norm_mode="except_first",
            num_grids=4,
            use_base_update=False
        ).to(device),
        "SechKAN5_norm2_rms": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="",
            norm2_type="rms",
            base_activation="silu",
            use_width=False,
            norm_mode="except_first",
            num_grids=4,
            use_base_update=False
        ).to(device),
       
        
    }'''


def train_and_eval(model, name, train_loader, val_loader, test_loader,
                   y_mean, y_std, epochs=100, lr=1e-3, patience=20, min_delta=1e-4):
    print(f"\nTraining {name}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.MSELoss()

    best_val_rmse = float('inf')
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    best_epoch, epochs_no_improve = 0, 0
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_train = time.perf_counter()
    
    for ep in range(epochs):
        model.train()
        batch_bar = tqdm(train_loader, desc=f"{name} | Epoch {ep+1:02d}/{epochs}", leave=False)
        
        for Xb, yb in batch_bar:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
             
            batch_bar.set_postfix(loss=f"{loss.item():.6f}")
        
        tr_rmse, tr_rel_l2, _ = evaluate(model, train_loader, criterion, y_mean, y_std)
        val_rmse, val_rel_l2, _ = evaluate(model, val_loader, criterion, y_mean, y_std)
        
        scheduler.step(val_rmse)
        
        improvement = best_val_rmse - val_rmse
        
        if improvement > min_delta:
            best_val_rmse = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = ep + 1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        print(f"Epoch {ep+1:02d} | Train RMSE={tr_rmse:.4f} - Rel L2={tr_rel_l2:.5f} | "
              f"Val RMSE={val_rmse:.4f} - Rel L2={val_rel_l2:.5f}")
              
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping after {patience} epochs without improvement.")
            break
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    train_time = time.perf_counter() - start_train
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    # Final Evaluation
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_test = time.perf_counter()
    
    train_rmse, train_rel_l2, _ = evaluate(model, train_loader, criterion, y_mean, y_std)
    val_rmse,   val_rel_l2,   _ = evaluate(model, val_loader,   criterion, y_mean, y_std)
    test_rmse,  test_rel_l2,  _ = evaluate(model, test_loader,  criterion, y_mean, y_std)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    test_time = time.perf_counter() - start_test
    
    params = count_params(model)
    
    print(f"\n[{name}] BEST MODEL (Epoch {best_epoch})")
    print(f"Train:\tRMSE={train_rmse:.6f}\tRel L2={train_rel_l2:.6f}")
    print(f"Val:\tRMSE={val_rmse:.6f}\tRel L2={val_rel_l2:.6f}")
    print(f"Test:\tRMSE={test_rmse:.6f}\tRel L2={test_rel_l2:.6f}")
    print(f"Train Time={train_time:.1f}s | Test Time={test_time:.2f}s | Params={params}")
    
    return {
        "train_rmse": train_rmse,
        "train_rel_l2": train_rel_l2,
        "val_rmse": val_rmse,
        "val_rel_l2": val_rel_l2,
        "test_rmse": test_rmse,
        "test_rel_l2": test_rel_l2,
        "train_time": train_time,
        "test_time": test_time,
        "total_time": train_time + test_time,
        "params": params,
        "best_epoch": best_epoch
    }


if __name__ == "__main__":
    #seeds = [0, 1, 2, 3, 4]
    seeds = [0, 1]
    
    print("Generating Navier-Stokes data...")
    X, Y, t, U, V = generate_navier_stokes(seed=42)
    
    print("Preparing dataset (time-based split)...")
    train_loader, val_loader, test_loader, y_mean, y_std = prepare_data(
        X, Y, t, U, V, batch_size=512, seed=42, split_mode="time"
    )
    
    # Model dimensions
    in_dim = 3
    out_dim = 2
    models_dict = build_models(in_dim, out_dim)
    
    results = {}
    
    for name, build in models_dict.items():
        print(f"\n{'='*50}")
        print(f"Evaluating model: {name}")
        print(f"{'='*50}")
        
        train_rmse_list = []
        train_rel_list = []
        val_rmse_list = []
        val_rel_list = []
        test_rmse_list = []
        test_rel_list = []
        train_time_list = []
        test_time_list = []
        param_count = None
        
        for seed in seeds:
            print(f"\n--- Seed: {seed} ---")
            set_seed(seed)
            model = build()
            
            res = train_and_eval(model, name, train_loader, val_loader, test_loader,
                               y_mean, y_std, epochs=100)
            
            train_rmse_list.append(res["train_rmse"])
            train_rel_list.append(res["train_rel_l2"])
            val_rmse_list.append(res["val_rmse"])
            val_rel_list.append(res["val_rel_l2"])
            test_rmse_list.append(res["test_rmse"])
            test_rel_list.append(res["test_rel_l2"])
            train_time_list.append(res["train_time"])
            test_time_list.append(res["test_time"])
            
            if param_count is None:
                param_count = res["params"]
            
            del model
            torch.cuda.empty_cache()
        
        results[name] = {
            "train_rmse_mean": np.mean(train_rmse_list),
            "train_rmse_std": np.std(train_rmse_list),

            "val_rmse_mean": np.mean(val_rmse_list),
            "val_rmse_std": np.std(val_rmse_list),

            "test_rmse_mean": np.mean(test_rmse_list),
            "test_rmse_std": np.std(test_rmse_list),

            "train_rel_l2_mean": np.mean(train_rel_list),
            "train_rel_l2_std": np.std(train_rel_list),

            "val_rel_l2_mean": np.mean(val_rel_list),
            "val_rel_l2_std": np.std(val_rel_list),

            "test_rel_l2_mean": np.mean(test_rel_list),
            "test_rel_l2_std": np.std(test_rel_list),

            "train_time_mean": np.mean(train_time_list),
            "train_time_std": np.std(train_time_list),

            "test_time_mean": np.mean(test_time_list),
            "test_time_std": np.std(test_time_list),

            "total_time_mean": np.mean(np.array(train_time_list) + np.array(test_time_list)),
            "total_time_std": np.std(np.array(train_time_list) + np.array(test_time_list)),

            "params": param_count,
            
            "train_rmse_list": train_rmse_list,
            "val_rmse_list": val_rmse_list,
            "test_rmse_list": test_rmse_list,
            
            "train_rel_list": train_rel_list,
            "val_rel_list": val_rel_list,
            "test_rel_list": test_rel_list,
            
            "train_time_list": train_time_list,
            "test_time_list": test_time_list,
            
        } 

    
    # Create folder if it does not exist
    os.makedirs("output/navier_stoke", exist_ok=True)

    # Create filename
    dt_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"train_{dt_string}.json"

    # Full path
    filepath = os.path.join("output", "navier_stoke", filename)

    # Save file
    with open(filepath, "w") as f:
        json.dump(results, f, indent=4, default=float)
            
    print("\n" + "="*50)
    print("Final results (Mean ± Std over 5 seeds)")
    print("="*50)

    for k, v in results.items():
        print(f"{k:12s} | "
              f"Train Rel L2={v['train_rel_l2_mean']:.5f}±{v['train_rel_l2_std']:.5f} | "
              f"Val Rel L2={v['val_rel_l2_mean']:.5f}±{v['val_rel_l2_std']:.5f} | "
              f"Test Rel L2={v['test_rel_l2_mean']:.5f}±{v['test_rel_l2_std']:.5f}")
        
        print(f"{' ':12s} | "
              f"Train RMSE={v['train_rmse_mean']:.6f}±{v['train_rmse_std']:.6f} | "
              f"Val RMSE={v['val_rmse_mean']:.6f}±{v['val_rmse_std']:.6f} | "
              f"Test RMSE={v['test_rmse_mean']:.6f}±{v['test_rmse_std']:.6f}")
        
        print(f"{' ':12s} | "
              f"Train Time={v['train_time_mean']:.2f}±{v['train_time_std']:.2f}s | "
              f"Test Time={v['test_time_mean']:.2f}±{v['test_time_std']:.2f}s | "
              f"Total={v['total_time_mean']:.2f}±{v['total_time_std']:.2f}s | "
              f"Params={v['params']}")
        
        print("-"*50)