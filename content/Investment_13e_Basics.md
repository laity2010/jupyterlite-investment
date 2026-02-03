# Investment 13e — Return & Risk Basics

This notebook runs entirely in the browser (JupyterLite).

- Mean return
- Volatility
- Sharpe ratio
- Skewness
- Excess kurtosis

```python
import numpy as np
from math import sqrt

# Example returns (replace with your own)
r = np.array([-0.1189, -0.2210, 0.2869, 0.1088, 0.0491])

freq = 1   # yearly
rf = 0.02

mean = r.mean() * freq
vol = r.std(ddof=0) * sqrt(freq)
sharpe = (mean - rf) / vol

def skewness(x):
    m = x.mean()
    s = x.std(ddof=0)
    return np.mean((x - m)**3) / s**3

def excess_kurtosis(x):
    m = x.mean()
    s = x.std(ddof=0)
    return np.mean((x - m)**4) / s**4 - 3

mean, vol, sharpe, skewness(r), excess_kurtosis(r)
```
