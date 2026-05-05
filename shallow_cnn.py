import torch
import torch.nn as nn

class ShallowCNN(nn.Module):
    def __init__(self, n_classes=4, n_channels=22, n_timepoints=1001):
        super(ShallowCNN, self).__init__()
        
        self.temporal_conv = nn.Conv2d(1, 40, (1, 25), padding=(0, 12))
        self.spatial_conv = nn.Conv2d(40, 40, (n_channels, 1))
        self.bn = nn.BatchNorm2d(40)
        self.pool = nn.AvgPool2d((1, 75), stride=(1, 15))
        self.dropout = nn.Dropout(0.5)
        
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_timepoints)
            x = self.temporal_conv(dummy)
            x = self.spatial_conv(x)
            x = self.bn(x)
            x = x ** 2
            x = self.pool(x)
            x = torch.log(torch.clamp(x, min=1e-7))
            flat_size = x.view(1, -1).size(1)
        
        self.classifier = nn.Linear(flat_size, n_classes)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.bn(x)
        x = x ** 2
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-7))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x