import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

from src.DataLoaders import DataLoader

# fetch dataset
auto_mpg = fetch_ucirepo(id=9)

# data (as pandas dataframes)
X: pd.DataFrame = auto_mpg.data.features
y: pd.DataFrame = auto_mpg.data.targets

# Replace NaN w/ mean of features
X_numpy = X.to_numpy()
mean = np.nanmean(X_numpy, axis=0)

AUTO_MPG_FEATURES = np.where(np.isnan(X_numpy), mean, X_numpy)
AUTO_MPG_LABELS = y.to_numpy()
