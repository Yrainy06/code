import re
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# -------------- Configuration --------------
LOG_PATH = Path("my_log.log")
PNG_OUT  = Path("train_loss_curve.png")

USE_LOG_Y = True          # Log scale for loss axis (recommended)
SMOOTH_FRAC = 0.02        # Fraction of points for rolling mean window (2%)
EMA_ALPHA = 0.02          # Exponential moving average smoothing factor
FIG_W, FIG_H = 6.0, 4.0   # Inches; adjust if your journal has strict sizes
DPI = 300                 # PNG export resolution
# -------------------------------------------

# Matplotlib styling for paper-quality figures (no explicit colors)
mpl.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "axes.linewidth": 0.8,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,
})

pattern = re.compile(r"batch\s+(\d+)\s*\[\s*loss\s+([^\]]+)\s*\]", re.IGNORECASE)

def parse_log(path: Path) -> pd.DataFrame:
    batches, losses = [], []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            try:
                b = int(m.group(1))
                # Allow scientific or normal notation; strip stray characters
                l = float(m.group(2).strip())
            except ValueError:
                continue
            # Keep all; we’ll dedupe later
            batches.append(b)
            losses.append(l)

    if not batches:
        raise RuntimeError("No matching lines found. Ensure lines look like: 'batch 123 [loss 4.56e-05]'")

    df = pd.DataFrame({"batch": batches, "loss": losses})
    # If a batch appears multiple times, keep the LAST occurrence
    df = df.drop_duplicates(subset="batch", keep="last")
    df = df.sort_values("batch").reset_index(drop=True)
    return df

def rolling_mean(y: np.ndarray, frac: float) -> np.ndarray:
    n = len(y)
    if n == 0:
        return y
    win = max(3, int(round(frac * n)))
    win = win + 1 if win % 2 == 0 else win  # use odd window
    if win >= n:
        win = max(3, n if n % 2 == 1 else n - 1)
    return pd.Series(y).rolling(window=win, center=True, min_periods=1).mean().to_numpy()

def ema(y: np.ndarray, alpha: float) -> np.ndarray:
    if len(y) == 0:
        return y
    out = np.empty_like(y, dtype=float)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1.0 - alpha) * out[i-1]
    return out

def main():
    df = parse_log(LOG_PATH)
    x = df["batch"].to_numpy()
    y = df["loss"].to_numpy()

    y_roll = rolling_mean(y, SMOOTH_FRAC)
    y_ema  = ema(y, EMA_ALPHA)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = plt.gca()

    # Raw curve
    ax.plot(x, y, linewidth=0.8, alpha=0.7, label="Raw")

    # Smoothed curves (choose one or keep both)
    ax.plot(x, y_roll, linewidth=1.5, label="Rolling mean")
    ax.plot(x, y_ema, linewidth=1.2, linestyle="--", label="EMA")

    if USE_LOG_Y:
        # Avoid errors if any non-positive values (shouldn't happen for loss, but guard anyway)
        ymin = np.nanmin(y[y > 0]) if np.any(y > 0) else 1e-12
        ax.set_yscale("log")
        ax.set_ylim(bottom=max(ymin * 0.5, 1e-12))

    ax.set_xlabel("Batch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss vs. Batch")
    ax.grid(True, which="both", linewidth=0.4, alpha=0.4)
    ax.minorticks_on()
    ax.legend(frameon=False)
    fig.tight_layout()

    fig.savefig(PNG_OUT, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()
