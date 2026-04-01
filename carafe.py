import torch
import torch.nn as nn
import torch.nn.functional as F

class CARAFE(nn.Module):
    """CARAFE: Content-Aware ReAssembly of Features (https://arxiv.org/abs/1905.02188)"""
    def __init__(self, in_channels, out_channels=None, kernel_size=3, up_factor=2):
        super().__init__()
        self.up_factor = up_factor  # 上采样倍数（如2倍）
        self.kernel_size = kernel_size  # 聚合核大小
        out_channels = out_channels or in_channels  # 输出通道数，默认与输入一致

        # 1. 压缩模块：减少通道数，降低计算量
        self.compression = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
        # 2. 编码模块：生成注意力权重（用于聚合特征）
        self.encoding = nn.Conv2d(
            in_channels // 4,
            (up_factor * kernel_size) ** 2,  # 输出通道数 = (上采样倍数×核大小)²
            kernel_size=kernel_size,
            padding=kernel_size // 2  # 保持特征图尺寸
        )
        # 3. 上采样卷积：最终输出目标尺寸
        self.output_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        N, C, H, W = x.shape  # 输入特征：(批量, 通道, 高, 宽)
        r = self.up_factor  # 上采样倍数
        k = self.kernel_size  # 聚合核大小

        # 步骤1：生成注意力权重（内容感知）
        compressed = self.compression(x)  # 压缩通道：(N, C/4, H, W)
        attention = self.encoding(compressed)  # 编码生成权重：(N, (rk)², H, W)
        attention = F.softmax(attention, dim=1)  # 权重归一化
        attention = attention.view(N, r*r, k*k, H, W)  # 重塑：(N, r², k², H, W)

        # 步骤2：特征展开（为聚合做准备）
        x_unfolded = F.unfold(
            x,
            kernel_size=k,
            padding=k//2,
            stride=1  # 滑动步长1，覆盖所有位置
        )  # 展开后：(N, C×k², H×W)
        x_unfolded = x_unfolded.view(N, C, k*k, H, W)  # 重塑：(N, C, k², H, W)

        # 步骤3：内容感知聚合（注意力加权）
        # 矩阵乘法：(N, C, k², H, W) × (N, r², k², H, W) → (N, C, r², H, W)
        out = torch.einsum('nckhw, nrkhw -> ncrhw', x_unfolded, attention)
        out = out.view(N, C*r*r, H, W)  # 重塑：(N, C×r², H, W)

        # 步骤4：上采样+输出卷积
        out = F.pixel_shuffle(out, r)  # 像素重排：(N, C, H×r, W×r)
        out = self.output_conv(out)  # 调整输出通道
        return out