

import numpy as np
import torch
from torch.utils.data import Dataset

class STEM4DDataset(Dataset):
    
    
    def __init__(self, noisy_data, window_size=3, bright_field_mask=None):
        """
        Args:
            noisy_data: 4D numpy array [scan_x, scan_y, det_x, det_y]
            window_size: size of neighbouring patterns (3 for 3x3)
            bright_field_mask: mask bright field area
        """
        self.data = noisy_data.astype(np.float32)
        self.scan_x, self.scan_y, self.det_x, self.det_y = self.data.shape
        self.window_size = window_size
        self.offset = window_size // 2
        self.bf_mask = bright_field_mask
        
        # Create a list of valid positions (excluding the edges)
        self.valid_positions = []
        for x in range(self.offset, self.scan_x - self.offset):
            for y in range(self.offset, self.scan_y - self.offset):
                self.valid_positions.append((x, y))
    
    def __len__(self):
        return len(self.valid_positions)
    
    def __getitem__(self, idx):
        x, y = self.valid_positions[idx]
        
        # Collect neighbouring patterns (8 neighbours for 3x3)
        neighbors = []
        for i in range(self.window_size):
            for j in range(self.window_size):
                # Skip the central pattern
                if i == self.offset and j == self.offset:
                    continue
                
                nx = x - self.offset + i
                ny = y - self.offset + j
                pattern = self.data[nx, ny]
                
                # Apply the bright field mask if available
                if self.bf_mask is not None:
                    pattern = pattern * self.bf_mask
                
                neighbors.append(pattern)
        
        # Neighbour stack as input channels
        input_tensor = torch.FloatTensor(np.stack(neighbors))
        
        # Target pattern
        target = self.data[x, y]
        if self.bf_mask is not None:
            target = target * self.bf_mask
        target_tensor = torch.FloatTensor(target).unsqueeze(0)
        
        return input_tensor, target_tensor, idx
