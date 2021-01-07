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

class Default (nn.Module):
    def __init__ (self):
        super (Default, self).__init__()
        
        self.model = nn.Sequential (
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
            nn.Linear (512, 1)
        )
        
    def forward (self, x):
        return self.model(x)

class Default_Extra (nn.Module):
    def __init__ (self):
        super (Default_Extra, self).__init__()
        
        self.model = nn.Sequential (
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
            nn.Linear (512, 512),
            nn.Linear (512, 1)
        )
        
    def forward (self, x):
        return self.model(x)

class Default_Boost (nn.Module):
    def __init__ (self):
        super (Default_Boost, self).__init__()
        
        self.model = nn.Sequential (
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
            nn.Linear (1024, 1024),
            nn.Linear (1024, 1)
        )
        
    def forward (self, x):
        return self.model(x)
    
class Default_Pad (nn.Module):
    """
    All conv layers: padding=same
    """
    def __init__ (self):
        super (Default_Pad, self).__init__()
        
        self.model = nn.Sequential (
            # block 1
            conv_3d_block (1, 64, kernel_size=5, padding=2),
            conv_3d_block (64, 128, kernel_size=3, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # block 2
            conv_3d_block (128, 256, kernel_size=3, padding=1),
            conv_3d_block (256, 512, kernel_size=3, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # global pool
            nn.AdaptiveAvgPool3d(1),
            
            # linear layers
            nn.Flatten(),
            nn.Linear (512, 512),
            nn.Linear (512, 1)
        )
        
    def forward (self, x):
        return self.model(x)
    
class Default_Air (nn.Module):
    """
    1st conv layer: kernel_size=3 instead of 5
    """
    def __init__ (self):
        super (Default_Air, self).__init__()
        
        self.model = nn.Sequential (
            # block 1
            conv_3d_block (1, 64, kernel_size=3),
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
            nn.Linear (512, 1)
        )
        
    def forward (self, x):
        return self.model(x)

class Default_Pro (nn.Module):
    """
    3 conv blocks instead of 2
    All conv layers: padding=same 
    """
    def __init__ (self):
        super (Default_Pro, self).__init__()
        
        self.model = nn.Sequential (
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
            nn.Dropout (0.4), #added July 9th, 2020 (after job 1116643)
            nn.Linear (512, 1)
        )
        
    def forward (self, x):
        return self.model(x)

class Default_Pro_Max (nn.Module):
    """
    3 conv blocks instead of 2
    All conv layers: padding=same 
    """
    def __init__ (self):
        super (Default_Pro_Max, self).__init__()
        
        self.model = nn.Sequential (
            # block 1
            conv_3d_block (1, 64, kernel_size=3, padding=1),
            conv_3d_block (64, 128, kernel_size=3, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # block 2
            conv_3d_block (128, 256, kernel_size=3, padding=1),
            conv_3d_block (256, 512, kernel_size=3, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # block 3
            conv_3d_block (512, 1024, kernel_size=3, padding=1),
            conv_3d_block (1024, 1024, kernel_size=3, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # global pool
            nn.AdaptiveAvgPool3d(1),
            
            # linear layers
            nn.Flatten(),
            nn.Linear (1024, 1024),
            nn.Linear (1024, 1)
        )
        
    def forward (self, x):
        return self.model(x)
    
class Default_GN (nn.Module):
    """
    Group normalization instead of batch norm
    """
    def __init__ (self, num_groups=8):
        super (Default_GN, self).__init__()
        
        self.model = nn.Sequential (
            # block 1
            conv_3d_block (1, 64, norm='gn', kernel_size=5),
            conv_3d_block (64, 128, norm='gn', kernel_size=3),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # block 2
            conv_3d_block (128, 256, norm='gn', kernel_size=3),
            conv_3d_block  (256, 512, norm='gn', kernel_size=3),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # global pool
            nn.AdaptiveAvgPool3d(1),
            
            # linear layers
            nn.Flatten(),
            nn.Linear (512, 512),
            nn.Linear (512, 1)
        )
        
    def forward (self, x):
        return self.model(x)

class Default_3X (nn.Module):
    """
    3 conv layers per conv block (instead of 2)
    """
    def __init__ (self):
        super (Default_3X, self).__init__()
        
        self.model = nn.Sequential (
            # block 1
            conv_3d_block (1, 64, kernel_size=5),
            conv_3d_block (64, 128, kernel_size=3),
            conv_3d_block (128, 128, kernel_size=3),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # block 2
            conv_3d_block (128, 256, kernel_size=3),
            conv_3d_block (256, 256, kernel_size=3),
            conv_3d_block (256, 512, kernel_size=3),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # global pool
            nn.AdaptiveAvgPool3d(1),
            
            # linear layers
            nn.Flatten(),
            nn.Linear (512, 512),
            nn.Linear (512, 1)
        )
        
    def forward (self, x):
        return self.model(x)

class Default_XL (nn.Module):
    """
    3 conv layers per conv block (instead of 2)
    3 conv blocks instead of 2
    """
    def __init__ (self):
        super (Default_XL, self).__init__()
        
        self.model = nn.Sequential (
            # block 1
            conv_3d_block (1, 64, kernel_size=3, padding=1),
            conv_3d_block (64, 128, kernel_size=3, padding=1),
            conv_3d_block (128, 128, kernel_size=3, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # block 2
            conv_3d_block (128, 256, kernel_size=3, padding=1),
            conv_3d_block (256, 256, kernel_size=3, padding=1),
            conv_3d_block (256, 512, kernel_size=3, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # block 3
            conv_3d_block (512, 512, kernel_size=3, padding=1),
            conv_3d_block (512, 1024, kernel_size=3, padding=1),
            conv_3d_block (1024, 1024, kernel_size=3, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # global pool
            nn.AdaptiveAvgPool3d(1),
            
            # linear layers
            nn.Flatten(),
            nn.Linear (1024, 1024),
            nn.Dropout (0.4), #added July 9th, 2020 (after job 1116643)
            nn.Linear (1024, 1)
        )
        
    def forward (self, x):
        return self.model(x)
