
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNAutoencoder(nn.Module):
    
    def __init__(self, in_channels=8, base_features=16):
        super().__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_features, base_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_features),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_features, base_features * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_features * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_features * 2, base_features * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_features * 2),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_features * 2, base_features * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_features * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_features * 4, base_features * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_features * 4),
            nn.ReLU(inplace=True)
        )
        
        # Bottleneck 
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_features * 4, base_features * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_features * 8),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.dec3 = nn.Sequential(
            nn.Conv2d(base_features * 8, base_features * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_features * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_features * 4, base_features * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_features * 4),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(base_features * 4, base_features * 2, 
                                      kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_features * 2, base_features * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_features * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_features * 2, base_features * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_features * 2),
            nn.ReLU(inplace=True)
        )
        
        self.up1 = nn.ConvTranspose2d(base_features * 2, base_features, 
                                      kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_features, base_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_features, base_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_features),
            nn.ReLU(inplace=True)
        )
        
        # Final layer
        self.final = nn.Conv2d(base_features, 1, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        
        # Bottleneck
        bottleneck = self.bottleneck(enc3)
        
        # Decoder 
        dec3 = self.dec3(bottleneck)
        dec2 = self.dec2(self.up2(dec3))
        dec1 = self.dec1(self.up1(dec2))
        
        # Output
        output = self.final(dec1)
        
       
        output = torch.clamp(output, min=0.0)
        
        return output