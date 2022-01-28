import torch
import numpy as np
from scipy import spatial

class Dataset(torch.utils.data.Dataset):
    def __init__(self, pts, knn, transform=None):        
        self.transform = transform
        self.pts = pts.astype(np.float32)

    def __len__(self):
        return len(self.pts)

    def __getitem__(self, idx):
        p = self.pts[idx]

        if self.transform:
            p = self.transform(p)

        return p