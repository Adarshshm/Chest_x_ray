import torch
import torch.nn as nn
import torch.nn.functional as F

class TBXNet(nn.Module):
    """
    TBXNet Architecture implementation featuring sequential convolutional blocks,
    max pooling, specialized feature fusion layers, and a dense classification head.
    """
    def __init__(self, num_classes=4):
        super(TBXNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.align1 = nn.AdaptiveAvgPool2d((16, 16))
        self.align2 = nn.AdaptiveAvgPool2d((16, 16))
        self.align3 = nn.AdaptiveAvgPool2d((16, 16))
        
        self.fusion_conv = nn.Conv2d(224, 128, kernel_size=1)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.dropout = nn.Dropout(p=0.3)
        self.fc_out = nn.Linear(512, num_classes)

    def forward(self, x):
        """
        Executes the forward propagation pass across convolutional layers,
        feature map alignment and fusion, dropout regularization, and classification.
        """
        x1 = self.pool1(F.relu(self.conv1(x)))
        x2 = self.pool2(F.relu(self.conv2(x1)))
        x3 = self.pool3(F.relu(self.conv3(x2)))
        
        f1 = self.align1(x1)
        f2 = self.align2(x2)
        f3 = self.align3(x3)
        
        fused = torch.cat([f1, f2, f3], dim=1)
        fused = F.relu(self.fusion_conv(fused))
        
        out = self.flatten(fused)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        logits = self.fc_out(out)
        
        return logits