# Файл: src/dataset.py

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
import random

class STEM4DDataset(Dataset):
    """Dataset для деноизинга 4D STEM данных"""
    
    def __init__(self, 
                 noisy_data: np.ndarray,
                 clean_data: Optional[np.ndarray] = None,
                 window_size: int = 3,
                 bright_field_mask: Optional[np.ndarray] = None,
                 transform=None):
        """
        Args:
            noisy_data: 4D массив (scan_x, scan_y, det_x, det_y)
            clean_data: 4D массив для supervised обучения (опционально)
            window_size: размер окна (3 = 3x3 соседей)
            bright_field_mask: маска для bright field региона
            transform: дополнительные преобразования
        """
        self.noisy_data = noisy_data
        self.clean_data = clean_data
        self.window_size = window_size
        self.offset = window_size // 2
        self.bright_field_mask = bright_field_mask
        self.transform = transform
        
        # Размеры
        self.scan_x, self.scan_y, self.det_x, self.det_y = noisy_data.shape
        
        # Валидные позиции (исключая края)
        self.valid_positions = []
        for x in range(self.offset, self.scan_x - self.offset):
            for y in range(self.offset, self.scan_y - self.offset):
                self.valid_positions.append((x, y))
    
    def __len__(self):
        return len(self.valid_positions)
    
    def __getitem__(self, idx):
        """Получить данные для обучения"""
        center_x, center_y = self.valid_positions[idx]
        
        # Собираем соседние паттерны
        neighbors = []
        for i in range(self.window_size):
            for j in range(self.window_size):
                if i == self.offset and j == self.offset:
                    continue  # Пропускаем центральный
                
                x = center_x - self.offset + i
                y = center_y - self.offset + j
                
                pattern = self.noisy_data[x, y]
                if self.bright_field_mask is not None:
                    pattern = pattern * self.bright_field_mask
                
                neighbors.append(pattern)
        
        # Stack соседей как каналы
        input_data = np.stack(neighbors, axis=0)  # (8, det_x, det_y)
        
        # Целевой паттерн (центральный)
        if self.clean_data is not None:
            target = self.clean_data[center_x, center_y]
        else:
            target = self.noisy_data[center_x, center_y]
        
        if self.bright_field_mask is not None:
            target = target * self.bright_field_mask
        
        # Преобразуем в тензоры
        input_tensor = torch.FloatTensor(input_data)
        target_tensor = torch.FloatTensor(target).unsqueeze(0)  # (1, det_x, det_y)
        
        # Дополнительная информация
        info = {
            'position': (center_x, center_y),
            'mean_intensity': float(target.mean())
        }
        
        return input_tensor, target_tensor, info

def create_data_loaders(noisy_data, clean_data=None, 
                       batch_size=32, val_split=0.1, 
                       bright_field_mask=None, num_workers=0):
    """Создать DataLoader для обучения и валидации"""
    
    dataset = STEM4DDataset(noisy_data, clean_data, 
                           bright_field_mask=bright_field_mask)
    
    # Разделение на train/val
    n_samples = len(dataset)
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val], 
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader