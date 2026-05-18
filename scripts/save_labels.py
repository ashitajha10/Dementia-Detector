import pandas as pd
import numpy as np

# Load your processed CSV
df = pd.read_csv("data/preprocessed_data.csv")

# Extract labels
labels = df["Label"].values

# Save as .npy
import os
os.makedirs("features", exist_ok=True)
np.save("features/labels.npy", labels)


print(" Labels saved as features/labels.npy")
