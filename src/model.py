# Файл: src/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Базовый блок свертки для U-Net"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class UNetDenoiser(nn.Module):
    """U-Net для деноизинга 4D STEM данных
    
    Адаптирован для работы на CPU с небольшими изображениями (48x48)
    """
    def __init__(self, in_channels=8, base_features=16):
        """
        Args:
            in_channels: количество входных каналов (8 соседних паттернов)
            base_features: базовое количество фильтров (меньше для CPU)
        """
        super().__init__()
        
        # Encoder (путь вниз)
        self.enc1 = ConvBlock(in_channels, base_features)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = ConvBlock(base_features, base_features * 2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = ConvBlock(base_features * 2, base_features * 4)
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck (самый нижний уровень)
        # Для 48x48 после 3 пулингов будет 6x6
        self.bottleneck = ConvBlock(base_features * 4, base_features * 8)
        
        # Decoder (путь вверх)
        self.up3 = nn.ConvTranspose2d(base_features * 8, base_features * 4, 
                                      kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_features * 8, base_features * 4)
        
        self.up2 = nn.ConvTranspose2d(base_features * 4, base_features * 2, 
                                      kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_features * 4, base_features * 2)
        
        self.up1 = nn.ConvTranspose2d(base_features * 2, base_features, 
                                      kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_features * 2, base_features)
        
        # Финальный слой
        self.final = nn.Conv2d(base_features, 1, kernel_size=1)
        
        # Активация для положительных значений
        self.output_activation = nn.ReLU()
        
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool3(enc3))
        
        # Decoder с skip connections
        dec3 = self.up3(bottleneck)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.up2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.up1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        # Выход
        output = self.final(dec1)
        output = self.output_activation(output)
        
        return output
    
    def count_parameters(self):
        """Подсчет параметров модели"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# Упрощенная версия для быстрых экспериментов на CPU
class SimpleDenoisingCNN(nn.Module):
    """Простая CNN для деноизинга (быстрее для CPU)"""
    def __init__(self, in_channels=8):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        return x