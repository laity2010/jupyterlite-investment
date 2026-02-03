# Investment 13e — Return & Risk Basics

This notebook runs entirely in the browser (JupyterLite).

- Mean return
- Volatility
- Sharpe ratio
- Skewness
- Excess kurtosis

```python
import numpy as np
import pandas as pd
from math import sqrt

# === 1) Paste your data here ===
# Option A: prices (recommended). One column. Newest last.
prices_text = """
100
90
70
120
133
""".strip()

# Option B: returns. One column. Newest last.
returns_text = ""  # leave empty if using prices

# === 2) Settings ===
freq = 1          # 252 daily, 12 monthly, 1 yearly
rf_annual = 0.02  # annual risk-free rate

# === helpers ===
def parse_col(text: str) -> np.ndarray:
    lines = [x.strip() for x in text.splitlines() if x.strip() != ""]
    return np.array([float(x) for x in lines], dtype=float)

def skewness(x):
    m = x.mean()
    s = x.std(ddof=0)
    return np.mean((x - m)**3) / s**3

def excess_kurtosis(x):
    m = x.mean()
    s = x.std(ddof=0)
    return np.mean((x - m)**4) / s**4 - 3

def max_drawdown(prices: np.ndarray) -> float:
    peak = np.maximum.accumulate(prices)
    dd = prices / peak - 1.0
    return dd.min()

# === 3) Build returns ===
if returns_text.strip():
    r = parse_col(returns_text)
    # If you provide returns, we also create a synthetic price path for drawdown
    p = 100 * np.cumprod(1 + r)
else:
    p = parse_col(prices_text)
    r = p[1:] / p[:-1] - 1

# === 4) Core stats ===
mean_ann = r.mean() * freq
vol_ann  = r.std(ddof=0) * sqrt(freq)
sharpe   = (mean_ann - rf_annual) / vol_ann if vol_ann != 0 else float("nan")
sk       = skewness(r)
ek       = excess_kurtosis(r)
mdd      = max_drawdown(p)

# === 5) Show nicely ===
out = pd.DataFrame(
    {
        "Metric": ["Mean (ann)", "Vol (ann)", "Sharpe", "Skewness", "Excess Kurtosis", "Max Drawdown"],
        "Value":  [mean_ann,     vol_ann,     sharpe,   sk,         ek,               mdd],
    }
)

out
```
