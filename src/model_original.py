import torch
import torch.nn as nn

class conv_block(nn.Module):
    """Convolution block consisting of two layers Conv + BatchNorm + ReLU"""
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """Upsampling from ConvTranspose2d"""
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class U_Net(nn.Module):
    """Original U-Net architecture from the mcemtools repository"""
    def __init__(self, img_ch=8, output_ch=1, n_kernels=64, mask=None):
        super(U_Net, self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        self.Conv1 = conv_block(ch_in=img_ch, ch_out=n_kernels)
        self.Conv2 = conv_block(ch_in=n_kernels, ch_out=2*n_kernels)
        self.Conv3 = conv_block(ch_in=2*n_kernels, ch_out=4*n_kernels)
        self.Conv4 = conv_block(ch_in=4*n_kernels, ch_out=8*n_kernels)
        self.Conv5 = conv_block(ch_in=8*n_kernels, ch_out=16*n_kernels)

        # Decoder
        self.Up5 = up_conv(ch_in=16*n_kernels, ch_out=8*n_kernels)
        self.Up_conv5 = conv_block(ch_in=16*n_kernels, ch_out=8*n_kernels)

        self.Up4 = up_conv(ch_in=8*n_kernels, ch_out=4*n_kernels)
        self.Up_conv4 = conv_block(ch_in=8*n_kernels, ch_out=4*n_kernels)
        
        self.Up3 = up_conv(ch_in=4*n_kernels, ch_out=2*n_kernels)
        self.Up_conv3 = conv_block(ch_in=4*n_kernels, ch_out=2*n_kernels)
        
        self.Up2 = up_conv(ch_in=2*n_kernels, ch_out=n_kernels)
        self.Up_conv2 = conv_block(ch_in=2*n_kernels, ch_out=n_kernels)

        self.Conv_1x1 = nn.Conv2d(n_kernels, output_ch, kernel_size=1, stride=1, padding=0)

        
        self.mask = mask
        self.mu_exact = None  
        self.mu = None       
        self.PACBED = None   
        
        
        self.register_buffer('scale_factor', torch.ones(1))
        
        
        with torch.no_grad():
            self.Conv_1x1.weight.data.normal_(0.0, 0.01)
            if self.Conv_1x1.bias is not None:
                self.Conv_1x1.bias.data.zero_()
    
    def reset(self):
        
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.mask = self.mask
        self.mu_exact = None
        self.mu = None
        self.PACBED = None
    
    def forward(self, x, inds=None):
        # Encoder path
        x1 = self.Conv1(x)
        
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # Decoder path with skip connections
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        
        d1 = d1 * self.scale_factor
        
        
        d1 = d1 ** 2
        
        # Application of corrections as in the original
        for dim in range(d1.shape[0]):
            if self.PACBED is not None:
                d1[dim] *= self.PACBED
            if self.mu_exact is not None:
                d1[dim] /= d1[dim].sum()
                d1[dim] *= self.mu_exact[inds[dim]]
            elif self.mu is not None:
                d1[dim] *= self.mu[inds[dim]]
                
        if self.mask is not None:
            d1[:, :, self.mask == 0] = 0 
        
        return d1
