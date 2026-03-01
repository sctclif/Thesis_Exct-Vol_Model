import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Load the data
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'combined_sweep_data.npz')

data = np.load(file_path)
X = data['histograms']  # The histograms (Features)
Y = data['targets']     # The [E_val, V_val] (Labels)

# 2. Standardize (Crucial when you have multiple levels of variance)
X_scaled = StandardScaler().fit_transform(X)

# 3. Run PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 4. Visualise the "Mechanism Map"
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Plot A: Color by Excitability Level
sc1 = ax[0].scatter(X_pca[:, 0], X_pca[:, 1], c=Y[:, 0], cmap='viridis', edgecolors='k', alpha=0.8)
plt.colorbar(sc1, ax=ax[0], label='Excitability Amount ($E$)')
ax[0].set_title('PCA Space: Coded by Excitability')
ax[0].set_xlabel('Principal Component 1')
ax[0].set_ylabel('Principal Component 2')

# Plot B: Color by Volatility Level
sc2 = ax[1].scatter(X_pca[:, 0], X_pca[:, 1], c=Y[:, 1], cmap='plasma', edgecolors='k', alpha=0.8)
plt.colorbar(sc2, ax=ax[1], label='Volatility Amount ($V_{std}$)')
ax[1].set_title('PCA Space: Coded by Volatility')
ax[1].set_xlabel('Principal Component 1')

plt.tight_layout()
plt.show()

# 5. Optional: Print the "Importance" of each PC
print(f"PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% of the variance")
print(f"PC2 explains {pca.explained_variance_ratio_[1]*100:.1f}% of the variance")