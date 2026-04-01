import torch
import torch.nn as nn
import torch.nn.functional as F


class PolaLinearAttention(nn.Module):
    def __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1,
                 kernel_size=5, alpha=4):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim

        self.qg = nn.Linear(dim, 2 * dim, bias=qkv_bias)  # 输入dim，输出2*dim（正确，因为要拆成q和g）
        self.kv = nn.Linear(dim, 2 * dim, bias=qkv_bias)  # 同样用dim，输出2*dim（拆成k和v）
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)

        self.power = nn.Parameter(torch.zeros(size=(1, self.num_heads, 1, self.head_dim)))
        self.alpha = alpha

        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, num_patches // (sr_ratio * sr_ratio), dim)))
        print('Linear Attention sr_ratio{} f{} kernel{}'.
              format(sr_ratio, alpha, kernel_size))

    # 原错误的 forward 方法（需要手动传 H、W）
    # def forward(self, x, H, W):

    # 修复后的 forward 方法（自动从 x 提 H、W，不用手动传）
    def forward(self, x):
        # 新增：从输入特征 x 中提取高度 H 和宽度 W（x 形状是 [B, C, H, W]）
        B, C, H, W = x.shape  # 这行是关键！自动获取 H 和 W
        # 把特征图从 [B, C, H, W] 转成 [B, N, C]（N=H*W，适配注意力模块输入）
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # 形状转换：(B, C, H, W) → (B, H*W, C)

        # 下面的代码完全不变！
        q, g = self.qg(x).reshape(B, H * W, 2, C).unbind(2)  # 这里把 N 换成 H*W，和上面对应

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
        else:
            kv = self.kv(x).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]
        n = k.shape[1]

        k = k + self.positional_encoding
        kernel_function = nn.ReLU()

        scale = nn.Softplus()(self.scale)
        power = 1 + self.alpha * nn.functional.sigmoid(self.power)

        q = q / scale
        k = k / scale
        q = q.reshape(B, H * W, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()  # N 换成 H*W
        k = k.reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()
        v = v.reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()

        q_pos = kernel_function(q) ** power
        q_neg = kernel_function(-q) ** power
        k_pos = kernel_function(k) ** power
        k_neg = kernel_function(-k) ** power

        q_sim = torch.cat([q_pos, q_neg], dim=-1)
        q_opp = torch.cat([q_neg, q_pos], dim=-1)
        k = torch.cat([k_pos, k_neg], dim=-1)

        v1, v2 = torch.chunk(v, 2, dim=-1)

        z = 1 / (q_sim @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k.transpose(-2, -1) * (n ** -0.5)) @ (v1 * (n ** -0.5))
        x_sim = q_sim @ kv * z
        z = 1 / (q_opp @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k.transpose(-2, -1) * (n ** -0.5)) @ (v2 * (n ** -0.5))
        x_opp = q_opp @ kv * z

        x = torch.cat([x_sim, x_opp], dim=-1)
        x = x.transpose(1, 2).reshape(B, H * W, C)  # N 换成 H*W

        if self.sr_ratio > 1:
            v = nn.functional.interpolate(v.transpose(-2, -1).reshape(B * self.num_heads, -1, n), size=H * W,
                                          mode='linear').reshape(B, self.num_heads, -1, H * W).transpose(-2, -1)

        v = v.reshape(B * self.num_heads, H, W, -1).permute(0, 3, 1, 2)  # 用提取的 H、W
        v = self.dwc(v).reshape(B, C, H * W).permute(0, 2, 1)  # 用提取的 H、W
        x = x + v
        x = x * g

        x = self.proj(x)
        x = self.proj_drop(x)

        # 新增：把输出从 [B, H*W, C] 转回 [B, C, H, W]（适配 YOLO 后续层）
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)  # 形状转回：(B, H*W, C) → (B, C, H, W)

        return x