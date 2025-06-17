# Файл: src/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PoissonNLLLoss(nn.Module):
    """Poisson Negative Log-Likelihood Loss
    
    Специально для данных с шумом подсчета (counting noise)
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        
    def forward(self, prediction, target):
        """
        Args:
            prediction: предсказанные значения (положительные)
            target: целевые значения
        """
        # Избегаем log(0)
        prediction = prediction + self.eps
        
        # Poisson NLL: target * log(pred) - pred - log(target!)
        # Последний член константа, поэтому игнорируем
        loss = prediction - target * torch.log(prediction)
        
        return loss.mean()

class CombinedLoss(nn.Module):
    """Комбинированная loss функция"""
    def __init__(self, mse_weight=0.5, poisson_weight=0.5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.poisson = PoissonNLLLoss()
        self.mse_weight = mse_weight
        self.poisson_weight = poisson_weight
        
    def forward(self, prediction, target):
        mse_loss = self.mse(prediction, target)
        poisson_loss = self.poisson(prediction, target)
        
        total_loss = (self.mse_weight * mse_loss + 
                     self.poisson_weight * poisson_loss)
        
        return total_loss, mse_loss, poisson_loss

def calculate_metrics(prediction, target):
    """Вычислить метрики качества"""
    with torch.no_grad():
        # MSE
        mse = F.mse_loss(prediction, target)
        
        # PSNR (Peak Signal-to-Noise Ratio)
        max_val = target.max()
        psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
        
        # SSIM (упрощенная версия)
        # Для полной версии используйте torchmetrics
        pred_mean = prediction.mean()
        target_mean = target.mean()
        
        pred_std = prediction.std()
        target_std = target.std()
        
        covariance = ((prediction - pred_mean) * 
                     (target - target_mean)).mean()
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * pred_mean * target_mean + c1) * 
                (2 * covariance + c2)) / \
               ((pred_mean ** 2 + target_mean ** 2 + c1) * 
                (pred_std ** 2 + target_std ** 2 + c2))
        
        # Correlation coefficient
        correlation = torch.corrcoef(torch.stack([
            prediction.flatten(), 
            target.flatten()
        ]))[0, 1]
        
    return {
        'mse': mse.item(),
        'psnr': psnr.item(),
        'ssim': ssim.item(),
        'correlation': correlation.item() if not torch.isnan(correlation) else 0.0
    }