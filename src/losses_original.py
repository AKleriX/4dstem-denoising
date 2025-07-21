import torch
import torch.nn as nn
import torch.nn.functional as F

class STEM4D_PoissonLoss(nn.Module):
    """Poisson negative log-likelihood loss"""
    def __init__(self):
        super().__init__()
        
    def forward(self, predicted, target):
        # Avoid log(0)
        eps = 1e-8
        predicted = torch.clamp(predicted, min=eps)
        
        # Poisson NLL: target * log(predicted) - predicted - log(target!)
        # The last term is a constant and does not affect the gradient.
        loss = -torch.sum(target * torch.log(predicted) - predicted)
        
        return loss / predicted.numel()


class CombinedLoss(nn.Module):
    """Combined loss: Poisson + PACBED + STEM regularization"""
    def __init__(self, warmup_epochs=8, pacbed_weight=0.02, stem_weight=0.01):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        self.pacbed_weight = pacbed_weight
        self.stem_weight = stem_weight
        self.poisson_weight = 1.0 - pacbed_weight - stem_weight  # 0.97
        
        self.mse_loss = nn.MSELoss()
        self.poisson_loss = STEM4D_PoissonLoss()
        
        
        self.target_pacbed = None
        self.target_stem_sums = None
    
    def set_epoch(self, epoch):
        self.current_epoch = epoch
    
    def set_targets(self, target_pacbed, target_stem_sums):
        
        self.target_pacbed = target_pacbed
        self.target_stem_sums = target_stem_sums
        
    def forward(self, predicted, target, indices=None):
        batch_size = predicted.shape[0]
        
        # Main loss (MSE or Poisson)
        if self.current_epoch < self.warmup_epochs:
            main_loss = self.mse_loss(predicted, target)
        else:
            main_loss = self.poisson_loss(predicted, target)
        
        # PACBED regularization
        pacbed_loss = 0
        if self.target_pacbed is not None:
            pred_pacbed = predicted.mean(dim=0).squeeze()
            pacbed_loss = F.mse_loss(pred_pacbed, self.target_pacbed)
        
        # STEM regularization
        stem_loss = 0
        if self.target_stem_sums is not None and indices is not None:
            pred_stem_sums = predicted.sum(dim=(2, 3)).squeeze()
            target_sums = torch.stack([self.target_stem_sums[idx] for idx in indices])
            stem_loss = F.mse_loss(pred_stem_sums, target_sums)
        
        # Combined loss with weights
        total_loss = (self.poisson_weight * main_loss + 
                     self.pacbed_weight * pacbed_loss + 
                     self.stem_weight * stem_loss)
        
        return total_loss