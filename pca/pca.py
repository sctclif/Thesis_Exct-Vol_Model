import numpy as np
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
# Load both datasets
exc_path = os.path.join(script_dir, 'excitability_data_matrix.npy')
vol_path = os.path.join(script_dir, 'volatility_data_matrix.npy')

# 3. Load using the single path string
exc_data = np.load(exc_path)
vol_data = np.load(vol_path)

# Combine them into one matrix for PCA
# Total rows = Exc_seeds + Vol_seeds. Total columns = 100.
X = np.vstack([exc_data, vol_data])

# Create the "Answers" (Labels) for the decoder
# 0 for Excitability, 1 for Volatility
y = np.array([0] * len(exc_data) + [1] * len(vol_data))

# Run PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

import matplotlib.pyplot as plt

# Define bin centers for plotting (matching your np.histogram setup)
num_bins = 100
bins = np.linspace(-0.5, 1.5, num_bins)

# --- 1. RAW DATA: MEAN DRIFT SIGNATURES ---
plt.figure(figsize=(10, 5))
# Excitability Mean + Std Dev
plt.plot(bins, np.mean(exc_data, axis=0), label='Excitability Model', color='blue', lw=2)
plt.fill_between(bins, np.mean(exc_data, axis=0) - np.std(exc_data, axis=0), 
                 np.mean(exc_data, axis=0) + np.std(exc_data, axis=0), color='blue', alpha=0.1)

# Volatility Mean + Std Dev
plt.plot(bins, np.mean(vol_data, axis=0), label='Volatility Model', color='orange', lw=2)
plt.fill_between(bins, np.mean(vol_data, axis=0) - np.std(vol_data, axis=0), 
                 np.mean(vol_data, axis=0) + np.std(vol_data, axis=0), color='orange', alpha=0.1)

plt.title("Mean Drift Signatures (Raw Data Input)")
plt.xlabel(r"Scaled Correlation Change ($\Delta C$)")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(alpha=0.3)

# --- 2. THE PRINCIPAL VECTOR (PC1 LOADINGS) ---
plt.figure(figsize=(10, 4))
# This shows how the 'Principal Vector' weights each bin
plt.bar(bins, pca.components_[0], width=(bins[1]-bins[0]), color='purple', alpha=0.6)
plt.axhline(0, color='black', linewidth=1)
plt.title("The Principal Vector (PC1 Loadings)")
plt.xlabel(r"$\Delta C$ value")
plt.ylabel("Weight in PC1")
plt.grid(axis='y', alpha=0.3)

# --- 3. 2D SLICE: DATA POINTS + PC1 VECTOR OVERLAY ---
# We pick the two bins that vary the MOST according to PCA
idx1 = np.argmax(pca.components_[0]) # Bin with strongest positive weight
idx2 = np.argmin(pca.components_[0]) # Bin with strongest negative weight

plt.figure(figsize=(7, 6))
plt.scatter(exc_data[:, idx1], exc_data[:, idx2], color='blue', label='Excitability Seeds', alpha=0.6)
plt.scatter(vol_data[:, idx1], vol_data[:, idx2], color='orange', label='Volatility Seeds', alpha=0.6)

# Drawing the PC1 Vector projection on this 2D slice
mean_x = np.mean(X[:, idx1])
mean_y = np.mean(X[:, idx2])
scale = 0.5 # Length for visibility
plt.arrow(mean_x, mean_y, pca.components_[0, idx1]*scale, pca.components_[0, idx2]*scale, 
          color='red', width=0.005, head_width=0.02, label='PC1 Vector (Projected)')

plt.xlabel(r"Density at $\Delta C \approx {bins[idx1]:.2f}$")
plt.ylabel(r"Density at $\Delta C \approx {bins[idx2]:.2f}$")
plt.title("2D Data Slice & Principal Vector Alignment")
plt.legend()
plt.tight_layout()

plt.show()