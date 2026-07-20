import torch
import torch.nn as nn

class test_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,padding=1)
        self.bn1=nn.BatchNorm2d(16)
        self.conv2=nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,padding=1,stride=2)#16x16
        self.bn2=nn.BatchNorm2d(32)
        self.conv3=nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding=1)#16x16
        self.bn3=nn.BatchNorm2d(32)
        self.conv4=nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding=1)#16x16
        self.bn4=nn.BatchNorm2d(32)
        self.conv5=nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding=1)#16x16
        self.bn5=nn.BatchNorm2d(32)
        self.conv6=nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding=1)#16x16
        self.bn6=nn.BatchNorm2d(32)
        self.conv7=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1,stride=2)#8x8
        self.bn7=nn.BatchNorm2d(64)
        self.conv8=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1,stride=2)#4x4
        self.bn8=nn.BatchNorm2d(128)
        self.conv9=nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1,stride=2)#2x2
        self.bn9=nn.BatchNorm2d(128)
        self.conv10=nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1,stride=2)#1x1
        self.bn10=nn.BatchNorm2d(128)
        self.fc1=nn.Linear(128,32)
        self.fcbn1=nn.BatchNorm1d(32)
        self.fc2=nn.Linear(32,10)


    def forward(self,x):
    # ============ 卷积块 1 ============
        x = self.conv1(x)          # (batch, 16, 32, 32)
        x = self.bn1(x)            # 批归一化，形状不变
        x = torch.relu(x)          # 激活，形状不变
    
    # ============ 卷积块 2 ============
        x = self.conv2(x)          # (batch, 32, 16, 16)  ← 第一次下采样（stride=2）
        x = self.bn2(x)
        x = torch.relu(x)
    
    # ============ 卷积块 3 ============
        x = self.conv3(x)          # (batch, 32, 16, 16)  ← 保持尺寸
        x = self.bn3(x)
        x = torch.relu(x)
    
    # ============ 卷积块 4 ============
        x = self.conv4(x)          # (batch, 32, 16, 16)
        x = self.bn4(x)
        x = torch.relu(x)
    
    # ============ 卷积块 5 ============
        x = self.conv5(x)          # (batch, 32, 16, 16)
        x = self.bn5(x)
        x = torch.relu(x)
    
    # ============ 卷积块 6 ============
        x = self.conv6(x)          # (batch, 32, 16, 16)
        x = self.bn6(x)
        x = torch.relu(x)
    
    # ============ 卷积块 7 ============
        x = self.conv7(x)          # (batch, 64, 8, 8)   ← 第二次下采样
        x = self.bn7(x)
        x = torch.relu(x)
    
    # ============ 卷积块 8 ============
        x = self.conv8(x)          # (batch, 128, 4, 4)  ← 第三次下采样
        x = self.bn8(x)
        x = torch.relu(x)
    
    # ============ 卷积块 9 ============
        x = self.conv9(x)          # (batch, 128, 2, 2)  ← 第四次下采样
        x = self.bn9(x)
        x = torch.relu(x)
    
    # ============ 卷积块 10 ============
        x = self.conv10(x)         # (batch, 128, 1, 1)  ← 第五次下采样，压缩到 1x1
        x = self.bn10(x)
        x = torch.relu(x)
    
    # ============ 展平（关键步骤！） ============
    # 现在的形状是 (batch, 128, 1, 1)，需要拉直成 (batch, 128)
        x = x.view(x.size(0), -1)  # x.size(0) 是 batch_size，-1 自动计算剩余维度（这里就是 128）
    
    # ============ 全连接块 1 ============
        x = self.fc1(x)            # (batch, 32)
        x = self.fcbn1(x)          # 批归一化，形状不变
        x = torch.relu(x)          # 激活，形状不变
    
    # ============ 全连接块 2（输出层） ============
        x = self.fc2(x)            # (batch, 10)
    # 注意：这里不加 softmax，因为 CrossEntropyLoss 内部自带 softmax
    
        return x
