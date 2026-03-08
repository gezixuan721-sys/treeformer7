import torch
import torch.nn as nn


class MFM(nn.Module):
    """Modulation Fusion Module (two-input feature fusion).

    Copied from (CVPR2024)LEGM和MFM特征融合模块.py.
    输入: 两个形状相同的特征图 (B, C, H, W)
    输出: 融合后的特征图 (B, C, H, W)
    """

    def __init__(self, dim, height=2, reduction=8):
        super(MFM, self).__init__()

        self.height = height
        d = max(int(dim / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats1, in_feats2):
        in_feats = [in_feats1, in_feats2]
        B, C, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats * attn, dim=1)
        return out
