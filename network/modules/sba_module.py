import torch
import torch.nn as nn


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SBA(nn.Module):
    """Simplified Boundary-aware module with learnable boundary weight.

    This is copied from 34a. 引入可学习的边界权重机制.py and
    slightly refactored into a standalone module file.
    """

    def __init__(self, input_dim=64, output_dim=64):
        super(SBA, self).__init__()

        # 通道调整
        self.fc1 = nn.Conv2d(input_dim, input_dim // 2, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(input_dim, input_dim // 2, kernel_size=1, bias=False)

        # 特征增强
        self.d_in1 = BasicConv2d(input_dim // 2, input_dim // 2, kernel_size=1)
        self.d_in2 = BasicConv2d(input_dim // 2, input_dim // 2, kernel_size=1)

        # 输出卷积
        self.conv = nn.Sequential(
            BasicConv2d(input_dim, input_dim, 3, 1, 1),
            nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False)
        )

        # 可学习边界权重图
        self.boundary_weight_conv = nn.Conv2d(input_dim // 2, 1, kernel_size=3, padding=1, bias=False)

        self.Sigmoid = nn.Sigmoid()

    def forward(self, H_feature, L_feature):
        # 通道调整
        L_feature = self.fc1(L_feature)
        H_feature = self.fc2(H_feature)

        # 基础权重
        g_L_feature = self.Sigmoid(L_feature)
        g_H_feature = self.Sigmoid(H_feature)

        # 可学习边界权重图
        boundary_weight = self.boundary_weight_conv(L_feature)
        boundary_weight = self.Sigmoid(boundary_weight)

        # 1x1 卷积
        L_feature = self.d_in1(L_feature)
        H_feature = self.d_in2(H_feature)

        # 边界调节融合
        L_feature = L_feature + boundary_weight * (L_feature * g_L_feature)
        H_feature = H_feature + boundary_weight * (H_feature * g_H_feature)

        # 融合输出
        out = self.conv(torch.cat([H_feature, L_feature], dim=1))
        return out
