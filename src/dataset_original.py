# Файл: src/dataset_original.py

import numpy as np
import torch
from torch.utils.data import Dataset

class STEM4DDataset(Dataset):
    """Dataset для деноизинга 4D STEM с использованием соседних паттернов"""
    
    def __init__(self, noisy_data, window_size=3, bright_field_mask=None):
        """
        Args:
            noisy_data: 4D numpy array [scan_x, scan_y, det_x, det_y]
            window_size: размер окна соседних паттернов (3 для 3x3)
            bright_field_mask: маска bright field области
        """
        self.data = noisy_data.astype(np.float32)
        self.scan_x, self.scan_y, self.det_x, self.det_y = self.data.shape
        self.window_size = window_size
        self.offset = window_size // 2
        self.bf_mask = bright_field_mask
        
        # Создаем список валидных позиций (исключая края)
        self.valid_positions = []
        for x in range(self.offset, self.scan_x - self.offset):
            for y in range(self.offset, self.scan_y - self.offset):
                self.valid_positions.append((x, y))
    
    def __len__(self):
        return len(self.valid_positions)
    
    def __getitem__(self, idx):
        x, y = self.valid_positions[idx]
        
        # Собираем соседние паттерны (8 соседей для 3x3)
        neighbors = []
        for i in range(self.window_size):
            for j in range(self.window_size):
                # Пропускаем центральный паттерн
                if i == self.offset and j == self.offset:
                    continue
                
                nx = x - self.offset + i
                ny = y - self.offset + j
                pattern = self.data[nx, ny]
                
                # Применяем маску bright field если есть
                if self.bf_mask is not None:
                    pattern = pattern * self.bf_mask
                
                neighbors.append(pattern)
        
        # Stack соседей как каналы входа
        input_tensor = torch.FloatTensor(np.stack(neighbors))
        
        # Целевой паттерн
        target = self.data[x, y]
        if self.bf_mask is not None:
            target = target * self.bf_mask
        target_tensor = torch.FloatTensor(target).unsqueeze(0)
        
        return input_tensor, target_tensor, idx
