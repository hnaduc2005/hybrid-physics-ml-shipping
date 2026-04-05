"""
Physics-Informed Machine Learning Model
========================================
Task   : Predict vessel fuel consumption & CO2 emissions
Dataset: processed_data.csv  (AIS-based vessel survey data, Amazon region)
Author : NCKH Project

Key design decisions
--------------------
* Targets (fuel, CO2) span ~7-700 L  → train in log1p-space for stability
* Acceleration & heading_change absent from static survey → engineered to 0
* CO2 label = fuel × EF (emission factor) – consistent with physics
* Physics regulariser uses a *rank-preserving* formulation so unit mismatch
  does not dominate.  The penalty discourages predictions that violate the
  ordering implied by the Holtrop-Mennen resistance proxy (R ∝ A·V³).
* λ = 0.01 keeps physics as a soft constraint while data drives learning.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ─────────────────────────────────────────────────────────────────────────────
# Physics constants
# ─────────────────────────────────────────────────────────────────────────────
RHO    = 1000.0   # kg/m³  water density
CD     = 0.8      # drag coefficient
SFC    = 200.0    # g/kWh  specific fuel consumption
EF     = 3.2      # CO2 emission factor  (kg CO2 / kg fuel)
LAMBDA = 0.01     # physics loss weight  (soft constraint)

# ─────────────────────────────────────────────────────────────────────────────
# Column mapping  (CSV uses Portuguese column names)
# ─────────────────────────────────────────────────────────────────────────────
C_SPEED  = "VELOCIDADE"           # knots
C_LEN    = "COMPRIMENTO"          # m
C_WIDTH  = "BOCA"                 # m
C_DRAFT  = "CALADO"               # m
C_FUEL   = "CONSUMO_COMBUSTIVEL"  # litres  (total trip, survey-reported)

FEATURE_COLS = ["speed", "acceleration", "heading_change", "length", "width", "draft"]
TARGET_COLS  = ["fuel", "CO2"]

IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_img")


# ─────────────────────────────────────────────────────────────────────────────
# 1. load_data
# ─────────────────────────────────────────────────────────────────────────────
def load_data(csv_path: str):
    """
    Load processed_data.csv, engineer missing features, and return:
        X_train, X_test           – normalised float32 arrays
        y_train_log, y_test_log   – log1p-transformed targets  (training)
        y_test_raw                – original-scale targets      (evaluation)
        scaler                    – fitted StandardScaler for X
        feat_test_raw             – original-scale test features (physics)

    Feature engineering
    -------------------
    speed          ← VELOCIDADE  (knots)
    acceleration   ← 0.0  (no time-series available in static survey)
    heading_change ← 0.0  (no trajectory available in static survey)
    length         ← COMPRIMENTO  (m)
    width          ← BOCA         (m)
    draft          ← CALADO       (m)
    fuel           ← CONSUMO_COMBUSTIVEL  (L)
    CO2            ← fuel × EF   (derived label)
    """
    if not os.path.isabs(csv_path):
        base = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base, csv_path)

    df = pd.read_csv(csv_path)
    print(f"[load_data] Loaded {len(df)} rows × {len(df.columns)} columns")

    data = pd.DataFrame({
        "speed"         : df[C_SPEED].astype(float),
        "acceleration"  : 0.0,
        "heading_change": 0.0,
        "length"        : df[C_LEN].astype(float),
        "width"         : df[C_WIDTH].astype(float),
        "draft"         : df[C_DRAFT].astype(float),
        "fuel"          : df[C_FUEL].astype(float),
    })
    data["CO2"] = data["fuel"] * EF

    # Drop missing / non-positive targets
    data = data.dropna().query("fuel > 0 and speed > 0").reset_index(drop=True)
    print(f"[load_data] After cleaning: {len(data)} rows")
    print(f"[load_data] Fuel range  : {data['fuel'].min():.1f} – {data['fuel'].max():.1f} L")
    print(f"[load_data] Speed range : {data['speed'].min():.1f} – {data['speed'].max():.1f} knots")

    X_raw = data[FEATURE_COLS].values.astype(np.float32)
    y_raw = data[TARGET_COLS].values.astype(np.float32)

    # ── Normalise features ────────────────────────────────────────────────────
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw).astype(np.float32)

    # ── Log1p-transform targets (training only – invert for evaluation) ───────
    y_log = np.log1p(y_raw).astype(np.float32)

    # ── 80/20 split ───────────────────────────────────────────────────────────
    idx = np.arange(len(X_scaled))
    idx_tr, idx_te = train_test_split(idx, test_size=0.2, random_state=42)

    X_tr = X_scaled[idx_tr];  X_te = X_scaled[idx_te]
    y_tr = y_log[idx_tr];     y_te = y_log[idx_te]

    y_test_raw      = y_raw[idx_te]
    feat_test_raw   = X_raw[idx_te]

    print(f"[load_data] Train: {len(X_tr)}   Test: {len(X_te)}")
    return X_tr, X_te, y_tr, y_te, y_test_raw, feat_test_raw, scaler


# ─────────────────────────────────────────────────────────────────────────────
# 2. build_model
# ─────────────────────────────────────────────────────────────────────────────
def build_model(input_dim: int = 6, output_dim: int = 2) -> nn.Module:
    """
    Feed-forward neural network.
    Architecture: 6 → 128 → 64 → 32 → 2
    Outputs are in log1p-space (no final activation needed).
    """
    model = nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.BatchNorm1d(128),
        nn.LeakyReLU(0.1),
        nn.Dropout(0.2),

        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.LeakyReLU(0.1),
        nn.Dropout(0.1),

        nn.Linear(64, 32),
        nn.BatchNorm1d(32),
        nn.LeakyReLU(0.1),

        nn.Linear(32, output_dim),
        # No activation: outputs are in log-space (can be any real number)
    )

    # Xavier initialisation on linear layers
    for module in model:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    return model


# ─────────────────────────────────────────────────────────────────────────────
# 3. compute_physics_loss
# ─────────────────────────────────────────────────────────────────────────────
def compute_physics_loss(
    y_pred_log: torch.Tensor,
    batch_features: torch.Tensor,
    scaler: StandardScaler,
    device: torch.device
) -> torch.Tensor:
    """
    Rank-preserving physics regularisation.

    The survey labels (litres) and the physics proxy (g/h from engine power)
    share the same *ordering* even though they differ in units.  This loss
    penalises pairs of predictions that violate the ordering implied by:

        Fuel_physics ∝ A · V³     (R×V, where R ∝ A·V²)
        A = width × draft,   V = speed (m/s)

    Specifically, for all pairs (i, j) in the batch we penalise:

        max(0, sign(phys_j - phys_i) × (fuel_pred_i - fuel_pred_j))

    which is a relaxed Kendall-tau violation.  We approximate this cheaply
    with the correlation between the physics proxy and the predictions.

    Loss_physics = 1 - corr(fuel_pred, phys_proxy)
                 + 1 - corr(co2_pred,  phys_proxy × EF)

    Both terms are in [0, 2].  A perfect correlation → 0 physics loss.
    """
    # ── inverse-transform to physical units ──────────────────────────────────
    feat_np   = batch_features.detach().cpu().numpy()
    feat_inv  = scaler.inverse_transform(feat_np).astype(np.float32)
    feat_phys = torch.tensor(feat_inv, dtype=torch.float32, device=device)

    speed_kn = feat_phys[:, 0].clamp(min=1e-3)   # knots
    width    = feat_phys[:, 4].clamp(min=0.1)     # m
    draft    = feat_phys[:, 5].clamp(min=0.1)     # m

    V     = speed_kn * 0.514444          # knots → m/s
    A     = width * draft                # frontal area m²
    R     = 0.5 * RHO * CD * A * V ** 2 # resistance N
    P     = R * V                        # power W
    P_kw  = P / 1000.0                  # kW

    # Physics proxy (proportional to fuel consumption)
    phys_fuel = SFC * P_kw               # g/h  (ordinal proxy)
    phys_co2  = phys_fuel * EF

    # ── predictions in original scale (expm1 of log-output) ─────────────────
    pred_fuel = torch.expm1(y_pred_log[:, 0].clamp(min=-10, max=15))
    pred_co2  = torch.expm1(y_pred_log[:, 1].clamp(min=-10, max=15))

    # ── Pearson correlation loss ─────────────────────────────────────────────
    def pearson_corr_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        xm  = x - x.mean()
        ym  = y - y.mean()
        num = (xm * ym).sum()
        den = (xm.pow(2).sum().sqrt() * ym.pow(2).sum().sqrt()).clamp(min=eps)
        return 1.0 - num / den      # 0 = perfect positive correlation

    loss_fuel = pearson_corr_loss(pred_fuel, phys_fuel)
    loss_co2  = pearson_corr_loss(pred_co2,  phys_co2)

    return loss_fuel + loss_co2


# ─────────────────────────────────────────────────────────────────────────────
# 4. train_model
# ─────────────────────────────────────────────────────────────────────────────
def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train_log: np.ndarray,
    scaler: StandardScaler,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    lam: float = LAMBDA,
    device: torch.device = torch.device("cpu"),
) -> list:
    """
    Train with combined data-driven MSE loss (log-space) + physics loss.
    Returns per-epoch total loss history.
    """
    model.to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # Cosine annealing: smoothly reduces LR → prevents overshooting
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=epochs, eta_min=1e-5
    )
    mse_fn = nn.MSELoss()

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train_log, dtype=torch.float32)
    dataset = TensorDataset(X_t, y_t)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    history = []
    model.train()

    print(f"\n{'─'*68}")
    print(f"{'Epoch':>6}  {'MSE (log)':>12}  {'Phys Loss':>12}  {'Total':>12}  {'LR':>10}")
    print(f"{'─'*68}")

    for epoch in range(1, epochs + 1):
        epoch_mse  = 0.0
        epoch_phys = 0.0
        n_batches  = 0

        model.train()
        for X_b, y_b in loader:
            X_b = X_b.to(device)
            y_b = y_b.to(device)

            optimiser.zero_grad()
            y_pred = model(X_b)

            data_loss = mse_fn(y_pred, y_b)
            phys_loss = compute_physics_loss(y_pred, X_b, scaler, device)
            total     = data_loss + lam * phys_loss

            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimiser.step()

            epoch_mse  += data_loss.item()
            epoch_phys += phys_loss.item()
            n_batches  += 1

        scheduler.step()

        avg_mse  = epoch_mse  / n_batches
        avg_phys = epoch_phys / n_batches
        avg_tot  = avg_mse + lam * avg_phys
        history.append(avg_tot)

        if epoch == 1 or epoch % 10 == 0:
            lr_now = optimiser.param_groups[0]["lr"]
            print(f"{epoch:>6}  {avg_mse:>12.5f}  {avg_phys:>12.5f}"
                  f"  {avg_tot:>12.5f}  {lr_now:>10.2e}")

    print(f"{'─'*68}\n")
    return history


# ─────────────────────────────────────────────────────────────────────────────
# 5. evaluate_model
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(
    model: nn.Module,
    X_test: np.ndarray,
    y_test_log: np.ndarray,
    y_test_raw: np.ndarray,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """
    Evaluate on the test split.
    Predictions are made in log-space then inverted (expm1) for metrics.
    """
    model.eval()
    with torch.no_grad():
        X_t      = torch.tensor(X_test, dtype=torch.float32).to(device)
        pred_log = model(X_t).cpu().numpy()

    # Invert log1p transform
    pred_raw = np.expm1(pred_log).clip(min=0.0)

    fuel_true = y_test_raw[:, 0]
    co2_true  = y_test_raw[:, 1]
    fuel_pred = pred_raw[:, 0]
    co2_pred  = pred_raw[:, 1]

    results = {
        "fuel_mse"  : mean_squared_error(fuel_true, fuel_pred),
        "fuel_mae"  : mean_absolute_error(fuel_true, fuel_pred),
        "fuel_r2"   : r2_score(fuel_true, fuel_pred),
        "co2_mse"   : mean_squared_error(co2_true, co2_pred),
        "co2_mae"   : mean_absolute_error(co2_true, co2_pred),
        "co2_r2"    : r2_score(co2_true, co2_pred),
        "fuel_true" : fuel_true,
        "fuel_pred" : fuel_pred,
        "co2_true"  : co2_true,
        "co2_pred"  : co2_pred,
    }

    # Also compute log-space MSE for logging
    log_mse_fuel = mean_squared_error(y_test_log[:, 0], pred_log[:, 0])
    log_mse_co2  = mean_squared_error(y_test_log[:, 1], pred_log[:, 1])

    w = 20
    print("=" * 62)
    print("             MODEL EVALUATION RESULTS")
    print("=" * 62)
    print(f"{'Metric':<{w}} {'Fuel Consumption':>18} {'CO2 Emissions':>18}")
    print("-" * 62)
    print(f"{'MSE  (original scale)':<{w}} {results['fuel_mse']:>18.4f} {results['co2_mse']:>18.4f}")
    print(f"{'MAE  (original scale)':<{w}} {results['fuel_mae']:>18.4f} {results['co2_mae']:>18.4f}")
    print(f"{'R²   (original scale)':<{w}} {results['fuel_r2']:>18.4f} {results['co2_r2']:>18.4f}")
    print(f"{'MSE  (log1p scale)':<{w}} {log_mse_fuel:>18.6f} {log_mse_co2:>18.6f}")
    print("=" * 62)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 6. plot_results
# ─────────────────────────────────────────────────────────────────────────────
def plot_results(history: list, results: dict, save_dir: str) -> None:
    """
    Save three publication-quality plots:
      1.  model/train_img/loss_curve.png
      2.  model/train_img/fuel_prediction.png
      3.  model/train_img/co2_prediction.png
    """
    os.makedirs(save_dir, exist_ok=True)

    # ── Dark-mode colour palette ──────────────────────────────────────────────
    BG      = "#0d1117"
    PANEL   = "#161b22"
    GRID    = "#21262d"
    TEXT    = "#e6edf3"
    BLUE    = "#58a6ff"
    GREEN   = "#3fb950"
    CORAL   = "#f78166"
    YELLOW  = "#e3b341"

    plt.rcParams.update({
        "figure.facecolor"  : BG,
        "axes.facecolor"    : PANEL,
        "axes.edgecolor"    : GRID,
        "axes.labelcolor"   : TEXT,
        "axes.titlecolor"   : TEXT,
        "xtick.color"       : TEXT,
        "ytick.color"       : TEXT,
        "text.color"        : TEXT,
        "grid.color"        : GRID,
        "grid.linestyle"    : "--",
        "grid.alpha"        : 0.6,
        "legend.facecolor"  : PANEL,
        "legend.edgecolor"  : GRID,
        "legend.framealpha" : 0.8,
        "font.family"       : "DejaVu Sans",
        "font.size"         : 11,
    })

    # ── 1. Training Loss vs Epoch ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))

    epochs = np.arange(1, len(history) + 1)
    ax.plot(epochs, history, color=BLUE, linewidth=2.2, label="Total Loss", zorder=3)
    ax.fill_between(epochs, history, alpha=0.12, color=BLUE)

    # Moving average (window = 5)
    if len(history) >= 5:
        ma = pd.Series(history).rolling(5, min_periods=1).mean().values
        ax.plot(epochs, ma, color=YELLOW, linewidth=1.4,
                linestyle="--", label="Moving avg (5)", zorder=4)

    ax.set_title("Training Loss vs Epoch", fontsize=15, fontweight="bold", pad=14)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Total Loss  (MSE log-space + λ·Physics)", fontsize=12)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    p = os.path.join(save_dir, "loss_curve.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved → {p}")

    # ── Helper: scatter + identity line + stats box ───────────────────────────
    def scatter_pred_vs_actual(ax, y_true, y_pred, title, color, unit):
        ax.scatter(y_true, y_pred, c=color, alpha=0.55, s=26,
                   edgecolors="none", label="Test samples", zorder=3)

        # Perfect-fit diagonal
        vmin = max(0, min(y_true.min(), y_pred.min()) * 0.90)
        vmax = max(y_true.max(), y_pred.max()) * 1.08
        ax.plot([vmin, vmax], [vmin, vmax], color="#ffffff",
                linewidth=1.6, linestyle="--", label="Perfect fit", zorder=4)

        # Trend line (polyfit degree 1)
        coeffs = np.polyfit(y_true, y_pred, 1)
        xs = np.linspace(vmin, vmax, 200)
        ax.plot(xs, np.polyval(coeffs, xs), color=YELLOW,
                linewidth=1.4, linestyle="-", label="Trend", zorder=5)

        r2  = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        box_text = f"R²  = {r2:.4f}\nMAE = {mae:.2f} {unit}"
        ax.annotate(
            box_text,
            xy=(0.04, 0.93), xycoords="axes fraction",
            verticalalignment="top", fontsize=10, color=TEXT,
            bbox=dict(boxstyle="round,pad=0.45", fc=BG, ec=GRID, alpha=0.8),
        )

        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel(f"Actual {unit}", fontsize=11)
        ax.set_ylabel(f"Predicted {unit}", fontsize=11)
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
        ax.legend(fontsize=9)
        ax.grid(True)

    # ── 2. Fuel prediction ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 7))
    scatter_pred_vs_actual(
        ax,
        results["fuel_true"], results["fuel_pred"],
        "Predicted vs Actual – Fuel Consumption",
        GREEN, "Fuel (L)"
    )
    fig.tight_layout()
    p2 = os.path.join(save_dir, "fuel_prediction.png")
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved → {p2}")

    # ── 3. CO2 prediction ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 7))
    scatter_pred_vs_actual(
        ax,
        results["co2_true"], results["co2_pred"],
        "Predicted vs Actual – CO2 Emissions",
        CORAL, "CO2 (kg)"
    )
    fig.tight_layout()
    p3 = os.path.join(save_dir, "co2_prediction.png")
    fig.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved → {p3}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  Physics-Informed ML  |  device: {device}")
    print(f"{'='*55}\n")

    os.makedirs(IMG_DIR, exist_ok=True)

    # ── 1. Load & prepare data ────────────────────────────────────────────────
    (X_train, X_test,
     y_train_log, y_test_log,
     y_test_raw, feat_test_raw,
     scaler) = load_data("processed_data.csv")

    # ── 2. Build model ────────────────────────────────────────────────────────
    model = build_model(input_dim=len(FEATURE_COLS), output_dim=len(TARGET_COLS))
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[model] Architecture:\n{model}")
    print(f"[model] Trainable parameters: {param_count:,}\n")

    # ── 3. Train ──────────────────────────────────────────────────────────────
    history = train_model(
        model, X_train, y_train_log, scaler,
        epochs=100,
        batch_size=32,
        lr=1e-3,
        lam=LAMBDA,
        device=device,
    )

    # ── 4. Evaluate ───────────────────────────────────────────────────────────
    results = evaluate_model(model, X_test, y_test_log, y_test_raw, device)

    # ── 5. Visualise & save plots ─────────────────────────────────────────────
    plot_results(history, results, IMG_DIR)

    print(f"\n[done] All plots saved to: {IMG_DIR}")
    print(f"[done] Run complete.\n")
