import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import copy
import time
import os
import json
from datetime import datetime

from utils import *
from models import EfficientKAN, FastKAN, BSRBF_KAN, FasterKAN, ReLUKAN, SechKAN, MLP

from scipy.fft import fft2, ifft2, fftfreq
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

            sq_error_sum += torch.sum(diff ** 2).item()

            rel_num += torch.sum(diff ** 2).item()
            rel_den += torch.sum(yb ** 2).item()

            n_samples += Xb.size(0)

    rmse = np.sqrt(sq_error_sum / (n_samples * out_dim))
    rel_l2 = np.sqrt(rel_num / (rel_den + eps))
    avg_loss = total_loss / n_samples

    return rmse, rel_l2, avg_loss

    

def generate_shallow_water(n_x=64, n_y=64, n_t=60,
                           g=9.81, dt=5e-4, nu=2e-3, seed=42):
    np.random.seed(seed)
    
    # Grid
    x = np.linspace(0, 1, n_x, endpoint=False)
    y = np.linspace(0, 1, n_y, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")
    
    # Bathymetry
    b = (0.4 * np.exp(-80 * ((X - 0.2)**2 + (Y - 0.3)**2)) -
         0.35 * np.exp(-70 * ((X - 0.8)**2 + (Y - 0.7)**2)) +
         0.12 * np.sin(10*np.pi*X) * np.sin(9*np.pi*Y))
    
    # Initial conditions
    h = np.where(X < 0.5, 2.0, 0.8)
    h += 0.05 * np.sin(6*np.pi*X) * np.sin(6*np.pi*Y)
    h = np.clip(h, 0.5, 3.0)
    
    u = 0.2 * np.tanh(8*(Y - 0.5))
    v = -0.15 * np.tanh(8*(X - 0.5))
    
    hu = h * u
    hv = h * v
    
    # Spectral operators
    kx = 2*np.pi * fftfreq(n_x, d=1/n_x)
    ky = 2*np.pi * fftfreq(n_y, d=1/n_y)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    
    kx_max = np.max(np.abs(kx))
    ky_max = np.max(np.abs(ky))
    dealias_mask = ((np.abs(KX) < (2/3)*kx_max) & 
                   (np.abs(KY) < (2/3)*ky_max)).astype(float)

    def dx(f):
        return np.real(ifft2(1j * KX * fft2(f)))
    
    def dy(f):
        return np.real(ifft2(1j * KY * fft2(f)))
    
    def lap(f):
        return dx(dx(f)) + dy(dy(f))
    
    def dealias(f):
        return np.real(ifft2(fft2(f) * dealias_mask))

    def rhs(h, hu, hv):
        h = np.clip(h, 0.5, 3.0)
        u = hu / (h + 1e-8)
        v = hv / (h + 1e-8)
        
        hu_u = dealias(hu * u)
        hu_v = dealias(hu * v)
        hv_u = dealias(hv * u)
        hv_v = dealias(hv * v)
        
        h_t = -(dx(hu) + dy(hv))
        
        hu_t = -(dx(hu_u + 0.5 * g * h**2) + dy(hu_v)) - g * h * dx(b) + nu * lap(hu)
        hv_t = -(dx(hv_u) + dy(hv_v + 0.5 * g * h**2)) - g * h * dy(b) + nu * lap(hv)
        
        return h_t, hu_t, hv_t

    def rk4(h, hu, hv):
        def safe(h, hu, hv):
            h = np.clip(np.nan_to_num(h), 0.5, 3.0)
            hu = np.nan_to_num(hu)
            hv = np.nan_to_num(hv)
            return h, hu, hv
        
        k1 = rhs(*safe(h, hu, hv))
        k2 = rhs(*safe(h + 0.5*dt*k1[0], hu + 0.5*dt*k1[1], hv + 0.5*dt*k1[2]))
        k3 = rhs(*safe(h + 0.5*dt*k2[0], hu + 0.5*dt*k2[1], hv + 0.5*dt*k2[2]))
        k4 = rhs(*safe(h + dt*k3[0], hu + dt*k3[1], hv + dt*k3[2]))
        
        h_new = h + dt*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0]) / 6
        hu_new = hu + dt*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1]) / 6
        hv_new = hv + dt*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2]) / 6
        
        return safe(h_new, hu_new, hv_new)

    # Time integration
    Ht, Ut, Vt = [], [], []
    for _ in range(n_t):
        h, hu, hv = rk4(h, hu, hv)
        u = hu / (h + 1e-8)
        v = hv / (h + 1e-8)
        Ht.append(h.copy())
        Ut.append(u.copy())
        Vt.append(v.copy())
    
    t = np.arange(n_t) * dt
    return X, Y, t, np.array(Ht), np.array(Ut), np.array(Vt)

def prepare_data(X, Y, T, H, U, V, batch_size=256, seed=42, split_mode="shuffle"):

    np.random.seed(seed)
    torch.manual_seed(seed)

    n_t, n_x, n_y = H.shape

    xx, yy = X, Y

    
    n_train_t = int(0.7 * n_t)
    n_val_t   = int(0.15 * n_t)

    idx = np.arange(n_t)

    if split_mode == "shuffle":
        idx = np.random.permutation(idx)

    train_idx = idx[:n_train_t]
    val_idx   = idx[n_train_t:n_train_t + n_val_t]
    test_idx  = idx[n_train_t + n_val_t:]

    def build(t_idx):
        xs, ys = [], []

        for t in t_idx:
            tt = np.full_like(X, T[t], dtype=np.float32)

            x_t = np.stack([X, Y, tt], axis=-1).reshape(-1, 3).astype(np.float32)
            y_t = np.stack([H[t], U[t], V[t]], axis=-1).reshape(-1, 3).astype(np.float32)

            xs.append(x_t)
            ys.append(y_t)

        return np.vstack(xs), np.vstack(ys)

    tr_x, tr_y = build(train_idx)
    va_x, va_y = build(val_idx)
    te_x, te_y = build(test_idx)

    # torch
    tr_x = torch.from_numpy(tr_x)
    tr_y = torch.from_numpy(tr_y)
    va_x = torch.from_numpy(va_x)
    va_y = torch.from_numpy(va_y)
    te_x = torch.from_numpy(te_x)
    te_y = torch.from_numpy(te_y)

    # normalization
    x_m = tr_x.mean(0, keepdim=True)
    x_s = tr_x.std(0, keepdim=True) + 1e-6

    y_m = tr_y.mean(0, keepdim=True)
    y_s = tr_y.std(0, keepdim=True) + 1e-6

    tr_x = (tr_x - x_m) / x_s
    va_x = (va_x - x_m) / x_s
    te_x = (te_x - x_m) / x_s

    tr_y = (tr_y - y_m) / y_s
    va_y = (va_y - y_m) / y_s
    te_y = (te_y - y_m) / y_s

    pin = torch.cuda.is_available()

    train_loader = DataLoader(TensorDataset(tr_x, tr_y), batch_size=batch_size, shuffle=True, pin_memory=pin)
    val_loader   = DataLoader(TensorDataset(va_x, va_y), batch_size=batch_size*2, shuffle=False, pin_memory=pin)
    test_loader  = DataLoader(TensorDataset(te_x, te_y), batch_size=batch_size*2, shuffle=False, pin_memory=pin)

    print("=" * 50)
    print("DATASET (SHALLOW WATER)")
    print("=" * 50)
    print(f"Split mode: {split_mode}")
    print(f"Train/Val/Test: {len(tr_x):,} / {len(va_x):,} / {len(te_x):,}")
    print("dtype: float32")
    print("=" * 50)

    return train_loader, val_loader, test_loader, y_m.numpy(), y_s.numpy()


'''def build_models(in_dim, out_dim):
    return {
        "SechKAN2_norm1_layer": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="layer",
            norm2_type="",
            base_activation="silu",
            use_width=False,
            norm_mode="except_first",
            num_grids=4,
            use_base_update=False
        ).to(device),
        "SechKAN3_norm2_mm": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="",
            norm2_type="mm",
            base_activation="silu",
            use_width=False,
            norm_mode="except_first",
            num_grids=4,
            use_base_update=False
        ).to(device),
        "SechKAN4_no_norm": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="",
            norm2_type="",
            base_activation="silu",
            use_width=False,
            norm_mode="except_first",
            num_grids=4,
            use_base_update=False
        ).to(device),
        
    }'''

def build_models(in_dim, out_dim):
    return {
        "SechKAN1": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="layer",
            norm2_type="",
            base_activation="silu",
            use_width=False,
            norm_mode="except_first",
            num_grids=8,
            use_base_update=False
        ).to(device),
        "FastKAN": lambda: FastKAN([in_dim, 63, 63, out_dim], num_grids=8, use_layernorm = "").to(device),
        "MLP": lambda: MLP([in_dim, 192, 192, out_dim], base_activation="silu", norm_type = "").to(device),
        "ReLUKAN": lambda: ReLUKAN([in_dim, 66, 66, out_dim], grid=5, k=3, norm_type = "").to(device),   
        "BSRBF_KAN": lambda: BSRBF_KAN([in_dim, 64, 64, out_dim],
                                        grid_size=5, spline_order=3, norm_type = "").to(device),
        "FasterKAN": lambda: FasterKAN([in_dim, 68, 67, out_dim], num_grids=8, norm_type = "").to(device),
        "EfficientKAN": lambda: EfficientKAN([in_dim, 61, 60, out_dim],
                                              grid_size=5, spline_order=3).to(device),
        
    }
    
'''def build_models(in_dim, out_dim):
    return {
        "FastKAN": lambda: FastKAN([in_dim, 63, 63, out_dim], num_grids=8, use_layernorm = "").to(device),
        "MLP": lambda: MLP([in_dim, 196, 196, out_dim], base_activation="silu", norm_type = "").to(device),
        "ReLUKAN": lambda: ReLUKAN([in_dim, 66, 66, out_dim], grid=5, k=3, norm_type = "").to(device),   
        "BSRBF_KAN": lambda: BSRBF_KAN([in_dim, 64, 64, out_dim],
                                        grid_size=5, spline_order=3, norm_type = "").to(device),
        "FasterKAN": lambda: FasterKAN([in_dim, 68, 67, out_dim], num_grids=8, norm_type = "").to(device),
    }'''

# Ablation: num_grids (seed 0 & 1)
'''def build_models(in_dim, out_dim):
    return {
        "SechKAN1_num_grids_8": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="layer",
            norm2_type="",
            base_activation="silu",
            use_width=False,
            norm_mode="except_first",
            num_grids=8,
            use_base_update=False
        ).to(device),
        "SechKAN1_num_grids_2": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="layer",
            norm2_type="",
            base_activation="silu",
            use_width=False,
            norm_mode="except_first",
            num_grids=2,
            use_base_update=False
        ).to(device),
        "SechKAN1_num_grids_4": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="layer",
            norm2_type="",
            base_activation="silu",
            use_width=False,
            norm_mode="except_first",
            num_grids=4,
            use_base_update=False
        ).to(device),
        "SechKAN1_num_grids_16": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="layer",
            norm2_type="",
            base_activation="silu",
            use_width=False,
            norm_mode="except_first",
            num_grids=16,
            use_base_update=False
        ).to(device),
        "SechKAN1_num_grids_32": lambda: SechKAN(
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
    
# Ablation: activations (seed 0 & 1)
'''def build_models(in_dim, out_dim):
    return {
        "SechKAN1_silu": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="layer",
            norm2_type="",
            base_activation="silu",
            use_width=False,
            norm_mode="except_first",
            num_grids=8,
            use_base_update=False
        ).to(device),
        "SechKAN1_softplus": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="layer",
            norm2_type="",
            base_activation="softplus",
            use_width=False,
            norm_mode="except_first",
            num_grids=8,
            use_base_update=False
        ).to(device),
        "SechKAN1_sigmoid": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="layer",
            norm2_type="",
            base_activation="sigmoid",
            use_width=False,
            norm_mode="except_first",
            num_grids=8,
            use_base_update=False
        ).to(device),
        "SechKAN1_relu": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="layer",
            norm2_type="",
            base_activation="relu",
            use_width=False,
            norm_mode="except_first",
            num_grids=8,
            use_base_update=False
        ).to(device),
        "SechKAN1_leaky_relu": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="layer",
            norm2_type="",
            base_activation="leaky_relu",
            use_width=False,
            norm_mode="except_first",
            num_grids=8,
            use_base_update=False
        ).to(device),
        "SechKAN1_elu": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="layer",
            norm2_type="",
            base_activation="elu",
            use_width=False,
            norm_mode="except_first",
            num_grids=8,
            use_base_update=False
        ).to(device),
        "SechKAN1_gelu": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="layer",
            norm2_type="",
            base_activation="gelu",
            use_width=False,
            norm_mode="except_first",
            num_grids=8,
            use_base_update=False
        ).to(device),
        "SechKAN1_selu": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="layer",
            norm2_type="",
            base_activation="selu",
            use_width=False,
            norm_mode="except_first",
            num_grids=8,
            use_base_update=False
        ).to(device),
        "SechKAN1_tanh": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="layer",
            norm2_type="",
            base_activation="tanh",
            use_width=False,
            norm_mode="except_first",
            num_grids=8,
            use_base_update=False
        ).to(device),
        
    }'''

# Ablation: norms (seed 0 & 1)
'''def build_models(in_dim, out_dim):
    return {
        "SechKAN1_norm1_layer": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="layer",
            norm2_type="",
            base_activation="silu",
            use_width=False,
            norm_mode="except_first",
            num_grids=8,
            use_base_update=False
        ).to(device),
        "SechKAN1_norm1_batch": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="batch",
            norm2_type="",
            base_activation="silu",
            use_width=False,
            norm_mode="except_first",
            num_grids=8,
            use_base_update=False
        ).to(device),
        "SechKAN1_norm1_mm": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="mm",
            norm2_type="",
            base_activation="silu",
            use_width=False,
            norm_mode="except_first",
            num_grids=8,
            use_base_update=False
        ).to(device),
        "SechKAN1_norm1_rms": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="rms",
            norm2_type="",
            base_activation="silu",
            use_width=False,
            norm_mode="except_first",
            num_grids=8,
            use_base_update=False
        ).to(device),
        "SechKAN1_none": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="",
            norm2_type="",
            base_activation="silu",
            use_width=False,
            norm_mode="except_first",
            num_grids=8,
            use_base_update=False
        ).to(device),
        "SechKAN1_norm2_layer": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="",
            norm2_type="layer",
            base_activation="silu",
            use_width=False,
            norm_mode="except_first",
            num_grids=8,
            use_base_update=False
        ).to(device),
        "SechKAN1_norm2_batch": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="",
            norm2_type="batch",
            base_activation="silu",
            use_width=False,
            norm_mode="except_first",
            num_grids=8,
            use_base_update=False
        ).to(device),
        "SechKAN1_norm2_mm": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="",
            norm2_type="mm",
            base_activation="silu",
            use_width=False,
            norm_mode="except_first",
            num_grids=8,
            use_base_update=False
        ).to(device),
        "SechKAN1_norm2_rms": lambda: SechKAN(
            [in_dim, 192, 192, out_dim],
            norm1_type="",
            norm2_type="rms",
            base_activation="silu",
            use_width=False,
            norm_mode="except_first",
            num_grids=8,
            use_base_update=False
        ).to(device),
        
    }'''
    
    
def train_and_eval(model, name, train_loader, val_loader, test_loader, y_mean=None, y_std=None, epochs=200, lr=1e-3, patience=20, min_delta=1e-4):
    
    print(f"\nTraining {name}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.MSELoss()

    best_val_rmse = float('inf')
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_train = time.perf_counter()

    for ep in range(epochs):

        model.train()

        batch_bar = tqdm(train_loader, desc=f"{name} | Epoch {ep+1:03d}/{epochs}", leave=False)

        for Xb, yb in batch_bar:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(Xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            batch_bar.set_postfix(loss=f"{loss.item():.6f}")

        tr_rmse, tr_rel_l2, _ = evaluate(model, train_loader, criterion, y_mean, y_std)
        val_rmse, val_rel_l2, _ = evaluate(model, val_loader, criterion, y_mean, y_std)

        scheduler.step(val_rmse)

        improvement = best_val_rmse - val_rmse

        if improvement > min_delta:
            best_val_rmse = val_rmse
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = ep + 1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {ep+1:03d} | "
            f"LR={current_lr:.2e} | "
            f"Train RMSE={tr_rmse:.4f} Rel L2={tr_rel_l2:.5f} | "
            f"Val RMSE={val_rmse:.4f} Rel L2={val_rel_l2:.5f}"
        )

        if epochs_no_improve >= patience:

            print(
                f"\nEarly stopping triggered "
                f"after {patience} epochs without improvement."
            )

            break

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    train_time = time.perf_counter() - start_train

    if best_state is not None:
        model.load_state_dict(best_state)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_test = time.perf_counter()

    train_rmse, train_rel_l2, _ = evaluate(model, train_loader, criterion, y_mean, y_std)
    val_rmse, val_rel_l2, _ = evaluate(model, val_loader, criterion, y_mean, y_std)
    test_rmse, test_rel_l2, _ = evaluate(model, test_loader, criterion, y_mean, y_std)

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


# Main
if __name__ == "__main__":
    #seeds = [0, 1, 2, 3, 4]
    seeds = [0, 1]
    
    print("Generating Shallow Water data...")
    X, Y, t, H, U, V = generate_shallow_water()

    # Prepare
    print("Preparing dataset...")
    train_loader, val_loader, test_loader, y_mean, y_std = prepare_data(
        X, Y, t, H, U, V, batch_size=512, seed=42, split_mode="time"
    )

    # Model dimensions
    in_dim = 3   # (x, y, t)
    out_dim = 3  # (h, u, v)
    
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
    os.makedirs("output/shallow_water", exist_ok=True)

    # Create filename
    dt_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"train_{dt_string}.json"

    # Full path
    filepath = os.path.join("output", "shallow_water", filename)

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