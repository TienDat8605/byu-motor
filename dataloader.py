import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2

class FlagellarMotorDataset(Dataset):
    def __init__(self, labels_df, data_dir, cube_size=16):
        """
        Args:
            labels_df (DataFrame): DataFrame containing tomogram IDs and motor locations.
            data_dir (str): Path to the preprocessed 3D tomograms.
            cube_size (int): Number of slices per 3D input block.
        """
        self.labels_df = labels_df
        self.data_dir = data_dir
        self.cube_size = cube_size
        self.tomo_ids = labels_df["tomo_id"].unique()

    def __len__(self):
        return len(self.tomo_ids)

    def __getitem__(self, idx):
        tomo_id = self.tomo_ids[idx]
        tomo_path = os.path.join(self.data_dir, tomo_id)

        # Load all slices
        slice_files = sorted(os.listdir(tomo_path))
        slices = [cv2.imread(os.path.join(tomo_path, f), cv2.IMREAD_GRAYSCALE) for f in slice_files]
        tomo_tensor = np.stack(slices, axis=0)  # Shape: (Z, H, W)

        # Normalize to [0,1]
        tomo_tensor = tomo_tensor.astype(np.float32) / 255.0

        # Get motor locations for this tomogram
        motor_locs = self.labels_df[self.labels_df["tomo_id"] == tomo_id]

        # Create binary labels for slices (1 if motor exists, else 0)
        z_labels = np.zeros(tomo_tensor.shape[0], dtype=np.float32)
        for _, row in motor_locs.iterrows():
            z_index = int(row["Motor axis 0"])
            z_labels[z_index] = 1  # Mark motor location

        # --- Apply Sliding Window (Crop to Fixed Depth) ---
        z_size = tomo_tensor.shape[0]  # Actual depth
        if z_size > self.cube_size:
            # Randomly select a starting point for the cube
            start_idx = np.random.randint(0, z_size - self.cube_size + 1)
            tomo_tensor = tomo_tensor[start_idx:start_idx + self.cube_size, :, :]
            z_labels = z_labels[start_idx:start_idx + self.cube_size]
        else:
            # If smaller, pad with zeros
            pad_size = self.cube_size - z_size
            tomo_tensor = np.pad(tomo_tensor, ((0, pad_size), (0, 0), (0, 0)), mode='constant')
            z_labels = np.pad(z_labels, (0, pad_size), mode='constant')

        return torch.tensor(tomo_tensor).unsqueeze(0), torch.tensor(z_labels)

# train_labels = pd.read_csv("train_labels_resized.csv")
# train_dataset = FlagellarMotorDataset(train_labels, "train_enhanced/", cube_size=16)
# train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
