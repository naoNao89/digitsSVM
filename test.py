import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the train dataset
df_train = pd.read_csv('Datasets/train.csv')

# Split train data into X_train, Y_train
Y_train = df_train.iloc[:, 0]  # Labels
X_train = df_train.iloc[:, 1:]  # Features

# Convert Pandas DataFrame to NumPy array
X_train_np = X_train.to_numpy()

# Normalize data
X_train_np = X_train_np / 255.0

# Perform PCA
pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
X_pca = pca.fit_transform(X_train_np)

# Visualize original data
plt.figure(figsize=(12, 6))

# Plot original data points
plt.subplot(121)
plt.scatter(X_train_np[:, 0], X_train_np[:, 1], c=Y_train, cmap='viridis', alpha=0.5)
plt.title('Original Data Points')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Label')

# Visualize PCA-transformed data
plt.subplot(122)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=Y_train, cmap='viridis', alpha=0.5)
plt.title('PCA Transformed Data Points')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Label')

plt.tight_layout()
plt.show()
