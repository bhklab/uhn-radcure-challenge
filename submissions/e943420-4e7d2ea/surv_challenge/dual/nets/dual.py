"""
Variations of the default (baseline) model.
"""
import os
import torch
import torch.nn as nn

def conv_3d_block (in_c, out_c, act='lrelu', norm='bn', num_groups=8, *args, **kwargs):
    activations = nn.ModuleDict ([
        ['relu', nn.ReLU(inplace=True)],
        ['lrelu', nn.LeakyReLU(0.1, inplace=True)]
    ])
    
    normalizations = nn.ModuleDict ([
        ['bn', nn.BatchNorm3d(out_c)],
        ['gn', nn.GroupNorm(int(out_c/num_groups), out_c)]
    ])
    
    return nn.Sequential(
        nn.Conv3d(in_c, out_c, *args, **kwargs),
        normalizations[norm],
        activations[act]
    )

"""
specify these variables
"""
n_clin_vars = 39

class Dual_Base (nn.Module):
    def __init__ (self):
        super (Dual_Base, self).__init__()
        
        self.radiomics = nn.Sequential (
            # block 1
            conv_3d_block (1, 64, kernel_size=5),
            conv_3d_block (64, 128, kernel_size=3),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # block 2
            conv_3d_block (128, 256, kernel_size=3),
            conv_3d_block (256, 512, kernel_size=3),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # global pool
            nn.AdaptiveAvgPool3d(1),
            
            # linear layers
            nn.Flatten(),
            nn.Linear (512, 512),
            nn.Linear (512, 512)
        )
        
        self.fc = nn.Sequential (
            nn.Linear (512+n_clin_vars, 256), 
            nn.Linear (256, 256),
            nn.Linear (256, 1)
       )
        
    def forward (self, x):
        img, clin_var = x
        cnn = self.radiomics (img) # returns 512 
        latent_concat = torch.cat ((cnn, clin_var), dim=1) # concatenates tensors sizes of 512 + 32 = 544
        return self.fc (latent_concat)
        
class Dual_Pro (nn.Module):
    def __init__ (self):
        super (Dual_Pro, self).__init__()
        
        self.radiomics = nn.Sequential (
            # block 1
            conv_3d_block (1, 64, kernel_size=3, padding=1),
            conv_3d_block (64, 128, kernel_size=3, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # block 2
            conv_3d_block (128, 128, kernel_size=3, padding=1),
            conv_3d_block (128, 256, kernel_size=3, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # block 3
            conv_3d_block (256, 512, kernel_size=3, padding=1),
            conv_3d_block (512, 512, kernel_size=3, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # global pool
            nn.AdaptiveAvgPool3d(1),
            
            # linear layers
            nn.Flatten(),
            nn.Linear (512, 512),
            nn.Linear (512, 512)
        )
        
        self.fc = nn.Sequential (
            nn.Linear (512+n_clin_vars, 256), 
            nn.Linear (256, 256),
            nn.Linear (256, 1)
       )
        
    def forward (self, x):
        img, clin_var = x
        cnn = self.radiomics (img) # returns 512 
        latent_concat = torch.cat ((cnn, clin_var), dim=1) # concatenates tensors sizes of 512 + 32 = 544
        return self.fc (latent_concat)
    
class Dual_Test (nn.Module):
    def __init__ (self):
        super (Dual_Test, self).__init__()
        
        self.radiomics = nn.Sequential (
            # block 1
            conv_3d_block (1, 64, kernel_size=5),
            conv_3d_block (64, 128, kernel_size=3),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # block 2
            conv_3d_block (128, 256, kernel_size=3),
            conv_3d_block (256, 512, kernel_size=3),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # global pool
            nn.AdaptiveAvgPool3d(1),
            
            # linear layers
            nn.Flatten(),
            nn.Linear (512, 512),
        )
        
        self.fc = nn.Sequential (
            nn.Linear (512+n_clin_vars, 512), 
            nn.Linear (512, 1)
       )
        
    def forward (self, x):
        img, clin_var = x
        cnn = self.radiomics (img) # returns 512 
        latent_concat = torch.cat ((cnn, clin_var), dim=1) # concatenates tensors sizes of 512 + 32 = 544
        return self.fc (latent_concat)

class Dual_Wide (nn.Module):
    def __init__ (self):
        super (Dual_Hope, self).__init__()
        
        self.radiomics = nn.Sequential (
            # block 1
            conv_3d_block (1, 64, kernel_size=5),
            conv_3d_block (64, 128, kernel_size=3),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # block 2
            conv_3d_block (128, 256, kernel_size=3),
            conv_3d_block (256, 512, kernel_size=3),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # global pool
            nn.AdaptiveAvgPool3d(1),
            
            # linear layers
            nn.Flatten(),
            nn.Linear (512, 1024),
        )
        
        self.fc = nn.Sequential (
            nn.Linear (1024+n_clin_vars, 1024), 
            nn.Linear (1024, 1)
       )
        
    def forward (self, x):
        img, clin_var = x
        cnn = self.radiomics (img) # returns 512 
        latent_concat = torch.cat ((cnn, clin_var), dim=1) # concatenates tensors sizes of 512 + 32 = 544
        return self.fc (latent_concat)
    
class Dual_Lite (nn.Module):
    def __init__ (self):
        super (Dual_Lite, self).__init__()
        
        self.radiomics = nn.Sequential (
            # block 1
            conv_3d_block (1, 64, kernel_size=5),
            conv_3d_block (64, 128, kernel_size=3),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # block 2
            conv_3d_block (128, 256, kernel_size=3),
            conv_3d_block (256, 512, kernel_size=3),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # global pool
            nn.AdaptiveAvgPool3d(1),
            
            # linear layers
            nn.Flatten(),
        )
        
        self.fc = nn.Sequential (
            nn.Linear (512+n_clin_vars, 512), 
            nn.Dropout (0.4),
            nn.Linear (512, 1)
       )
        
    def forward (self, x):
        img, clin_var = x
        cnn = self.radiomics (img) # returns 512 
        latent_concat = torch.cat ((cnn, clin_var), dim=1) # concatenates tensors sizes of 512 + 32 = 544
        return self.fc (latent_concat)

class Dual_Drop0 (nn.Module):
    def __init__ (self):
        super (Dual_Drop0, self).__init__()
        
        self.radiomics = nn.Sequential (
            # block 1
            conv_3d_block (1, 64, kernel_size=5),
            conv_3d_block (64, 128, kernel_size=3),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # block 2
            conv_3d_block (128, 256, kernel_size=3),
            conv_3d_block (256, 512, kernel_size=3),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # global pool
            nn.AdaptiveAvgPool3d(1),
            
            # linear layers
            nn.Flatten(),
            nn.Linear (512, 512),
        )
        
        self.fc = nn.Sequential (
            nn.Linear (512+n_clin_vars, 512), 
            nn.Dropout (0.4),
            nn.Linear (512, 1)
       )
        
    def forward (self, x):
        img, clin_var = x
        cnn = self.radiomics (img) # returns 512 
        latent_concat = torch.cat ((cnn, clin_var), dim=1) # concatenates tensors sizes of 512 + 32 = 544
        return self.fc (latent_concat)
    
    
class Dual_Drop (nn.Module):
    def __init__ (self):
        super (Dual_Drop, self).__init__()
        
        self.radiomics = nn.Sequential (
            # block 1
            conv_3d_block (1, 64, kernel_size=5),
            conv_3d_block (64, 128, kernel_size=3),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # block 2
            conv_3d_block (128, 256, kernel_size=3),
            conv_3d_block (256, 512, kernel_size=3),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # global pool
            nn.AdaptiveAvgPool3d(1),
            
            # linear layers
            nn.Flatten(),
            nn.Linear (512, 1024),
        )
        
        self.fc = nn.Sequential (
            nn.Linear (1024+n_clin_vars, 1024), 
            nn.Dropout (0.4),
            nn.Linear (1024, 1)
       )
        
    def forward (self, x):
        img, clin_var = x
        cnn = self.radiomics (img) # returns 512 
        latent_concat = torch.cat ((cnn, clin_var), dim=1) # concatenates tensors sizes of 512 + 32 = 544
        return self.fc (latent_concat)