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

        # Load all slices and stack them into a 3D array
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
            z_labels[z_index] = 1  # Mark the motor location in the Z-dimension

        return torch.tensor(tomo_tensor).unsqueeze(0), torch.tensor(z_labels)

def custom_collate(batch):
    """Pad tomograms along depth (Z-axis) to match the largest depth in the batch."""
    tomos, labels = zip(*batch)  # Unpack batch

    # Find max depth (Z-axis size) in this batch
    max_depth = max(t.shape[1] for t in tomos)

    # Pad each tomogram along the Z-axis
    padded_tomos = [torch.nn.functional.pad(t, (0, 0, 0, 0, 0, max_depth - t.shape[1])) for t in tomos]

    # Stack into a single tensor
    tomo_tensor = torch.stack(padded_tomos)
    labels_tensor = torch.stack(labels)  # 🛠 Fix here: Stack labels properly

    return tomo_tensor, labels_tensor

train_labels = pd.read_csv("train_labels_resized.csv")
train_dataset = FlagellarMotorDataset(train_labels, "train_enhanced/", cube_size=16)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=custom_collate)
