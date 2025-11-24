import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)

class ZeroNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 128 Filters = High Intelligence
        # 10 Blocks = Deep Strategic Thinking
        self.filters = 128  
        self.blocks = 10    
        
        self.conv_input = nn.Conv2d(8, self.filters, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(self.filters)
        
        self.res_layers = nn.ModuleList([ResidualBlock(self.filters) for _ in range(self.blocks)])
        
        self.policy_conv = nn.Conv2d(self.filters, 32, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, 12)
        
        self.value_conv = nn.Conv2d(self.filters, 16, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(16)
        self.value_fc1 = nn.Linear(16 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn_input(self.conv_input(x)))
        for layer in self.res_layers:
            x = layer(x)
            
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(-1, 32 * 8 * 8)
        p = self.policy_fc(p)
        p = F.log_softmax(p, dim=1)
        
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(-1, 16 * 8 * 8)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        
        return p, v