# R. Quinlan. "Auto MPG," UCI Machine Learning Repository, 1993. [Online]. Available: https://doi.org/10.24432/C5859H.

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

# fetch dataset
auto_mpg = fetch_ucirepo(id=9)

# variable information
print(auto_mpg.variables)

# data (as pandas dataframes)
X: pd.DataFrame = auto_mpg.data.features
y: pd.DataFrame = auto_mpg.data.targets

# Replace NaN w/ mean of features
X_numpy = X.to_numpy()
mean = np.nanmean(X_numpy, axis=0)



def auto_mpg_normalize_features(features: np.ndarray) -> np.ndarray:
    return (features - np.nanmean(features, axis=0)) / np.nanstd(features, axis=0)

def auto_mpg_denormalize_labels(normed_labels: np.ndarray) -> np.ndarray:
    return np.multiply(normed_labels, label_std_dev) + label_mean

# assertion_features = np.allclose(AUTO_MLP_NORMALIZED_LABELS, normalize_features(AUTO_MPG_LABELS))
# assertion_labels = np.allclose(AUTO_MPG_LABELS, denormalize_labels(AUTO_MLP_NORMALIZED_LABELS))

AUTO_MPG_FEATURES = np.where(np.isnan(X_numpy), mean, X_numpy)
AUTO_MPG_LABELS = y.to_numpy()

# remove model year and origin (which we expect to be less important) from features: those are indices 5,6
AUTO_MPG_OMIT_FEATURES_EXPERIMENT = np.delete(AUTO_MPG_FEATURES, [5, 6], axis=1)
AUTO_MPG_NORMALIZED_OMIT_FEATURES_EXPERIMENT = auto_mpg_normalize_features(AUTO_MPG_OMIT_FEATURES_EXPERIMENT)

# Need nan version of numpy funcs b/c exists a single feature that contains a NaN
AUTO_MPG_NORMALIZED_FEATURES = (AUTO_MPG_FEATURES - np.nanmean(AUTO_MPG_FEATURES, axis=0)) / np.nanstd(AUTO_MPG_FEATURES, axis=0)

label_mean = np.nanmean(AUTO_MPG_LABELS, axis=0)
label_std_dev = np.nanstd(AUTO_MPG_LABELS, axis=0)
AUTO_MLP_NORMALIZED_LABELS = (AUTO_MPG_LABELS - label_mean) / label_std_dev

print("Auto MPG dataset loaded.")