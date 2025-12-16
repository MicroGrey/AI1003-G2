'''
Modified BSRN for Efficient SR
Improvements:
1. Pixel Attention (PA) replacing CCA for better spatial detail recovery.
2. SiLU activation for smoother gradient flow and no-parameter overhead.
3. Self-contained Upsampler implementation.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from basicsr.utils.registry import ARCH_REGISTRY

# =========================================================================
# 1. 基础组件 
# =========================================================================

class PixelShuffleDirect(nn.Module):
    def __init__(self, scale, num_feat, num_out_ch):
        super(PixelShuffleDirect, self).__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(num_feat, num_out_ch * scale * scale, kernel_size=3, padding=1),
            nn.PixelShuffle(scale)
        )
    def forward(self, x):
        return self.upsample(x)

class PixelAttention(nn.Module):
    def __init__(self, num_feat):
        super(PixelAttention, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 生成像素级权重图
        attn = self.conv1(x)
        attn = self.sigmoid(attn)
        return x * attn

# =========================================================================
# 2. 卷积算子
# =========================================================================

class BSConvU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_ln=False, bn_kwargs=None):
        super().__init__()
        self.pw = torch.nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.dw = torch.nn.Conv2d(
                out_channels, out_channels, kernel_size, stride, padding, dilation, 
                groups=out_channels, bias=bias, padding_mode=padding_mode
        )

    def forward(self, fea):
        fea = self.pw(fea)
        fea = self.dw(fea)
        return fea

class BSConvS(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True,
                 padding_mode="zeros", p=0.25, min_mid_channels=4, with_ln=False, bn_kwargs=None):
        super().__init__()
        mid_channels = min(in_channels, max(min_mid_channels, math.ceil(p * in_channels)))
        self.pw1 = torch.nn.Conv2d(in_channels, mid_channels, 1, 1, 0, bias=False)
        self.pw2 = torch.nn.Conv2d(mid_channels, out_channels, 1, 1, 0, bias=False)
        self.dw = torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, groups=out_channels, bias=bias)

    def forward(self, x):
        fea = self.pw1(x)
        fea = self.pw2(fea)
        fea = self.dw(fea)
        return fea

# =========================================================================
# 3. 核心模块 (ESA & ESDB)
# =========================================================================

class ESA(nn.Module):
    def __init__(self, num_feat=50, conv=nn.Conv2d, p=0.25):
        super(ESA, self).__init__()
        f = num_feat // 4
        BSConvS_kwargs = {'p': p} if conv.__name__ == 'BSConvS' else {}
        self.conv1 = nn.Conv2d(num_feat, f, 1)
        self.conv_f = nn.Conv2d(f, f, 1)
        self.maxPooling = nn.MaxPool2d(kernel_size=7, stride=3)
        self.conv_max = conv(f, f, kernel_size=3, **BSConvS_kwargs)
        self.conv2 = conv(f, f, 3, 2, 0)
        self.conv3 = conv(f, f, kernel_size=3, **BSConvS_kwargs)
        self.conv3_ = conv(f, f, kernel_size=3, **BSConvS_kwargs)
        self.conv4 = nn.Conv2d(f, num_feat, 1)
        self.sigmoid = nn.Sigmoid()
        # [改进] 使用 SiLU，无参数，性能好
        self.act = nn.SiLU()

    def forward(self, input):
        c1_ = (self.conv1(input))
        c1 = self.conv2(c1_)
        v_max = self.maxPooling(c1)
        v_range = self.act(self.conv_max(v_max))
        c3 = self.act(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (input.size(2), input.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4((c3 + cf))
        m = self.sigmoid(c4)
        return input * m

class ESDB(nn.Module):
    def __init__(self, in_channels, out_channels, conv=nn.Conv2d, p=0.25):
        super(ESDB, self).__init__()
        # 默认 padding=1
        kwargs = {'padding': 1} 
        if conv.__name__ == 'BSConvS':
            kwargs['p'] = p

        self.dc = self.distilled_channels = in_channels // 2
        self.rc = self.remaining_channels = in_channels

        self.c1_d = nn.Conv2d(in_channels, self.dc, 1)
        self.c1_r = conv(in_channels, self.rc, kernel_size=3, **kwargs)
        self.c2_d = nn.Conv2d(self.remaining_channels, self.dc, 1)
        self.c2_r = conv(self.remaining_channels, self.rc, kernel_size=3, **kwargs)
        self.c3_d = nn.Conv2d(self.remaining_channels, self.dc, 1)
        self.c3_r = conv(self.remaining_channels, self.rc, kernel_size=3, **kwargs)
        self.c4 = conv(self.remaining_channels, self.dc, kernel_size=3, **kwargs)
        
        # [改进] 使用 SiLU，彻底解决通道不匹配和参数共享问题
        self.act = nn.SiLU()

        self.c5 = nn.Conv2d(self.dc * 4, in_channels, 1)
        self.esa = ESA(in_channels, conv)
        self.pa = PixelAttention(in_channels)

    def forward(self, input):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = self.act(self.c1_r(input) + input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = self.act(self.c2_r(r_c1) + r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = self.act(self.c3_r(r_c2) + r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out = self.c5(out)
        
        # 双重注意力机制 (Dual Attention)
        out = self.esa(out) # 关注长距离和通道
        out = self.pa(out)  # 关注像素级细节
        
        return out + input

# =========================================================================
# 4. 主网络
# =========================================================================

@ARCH_REGISTRY.register()
class dual_atten_BSRN(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=64, num_block=8, num_out_ch=3, upscale=2,
                 conv='BSConvU', p=0.25):
        super(dual_atten_BSRN, self).__init__()
        kwargs = {'padding': 1}
        if conv == 'BSConvS':
            kwargs = {'p': p}
        
        if conv == 'BSConvU': self.conv = BSConvU
        elif conv == 'BSConvS': self.conv = BSConvS
        else: self.conv = nn.Conv2d
        
        # 1. 浅层特征提取
        self.fea_conv = self.conv(num_in_ch * 4, num_feat, kernel_size=3, **kwargs)

        # 2. 堆叠 Block
        self.blocks = nn.ModuleList()
        for _ in range(num_block):
            self.blocks.append(ESDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p))

        # 3. 特征融合
        self.c1 = nn.Conv2d(num_feat * num_block, num_feat, 1)
        
        # [改进] 尾部激活使用 SiLU
        self.act = nn.SiLU()

        self.c2 = self.conv(num_feat, num_feat, kernel_size=3, **kwargs)

        # 4. 上采样 (直接调用内部类)
        self.upsampler = PixelShuffleDirect(scale=upscale, num_feat=num_feat, num_out_ch=num_out_ch)

    def forward(self, input):
        # Input Replication
        input = torch.cat([input, input, input, input], dim=1)
        out_fea = self.fea_conv(input)
        
        distilled_features = []
        x = out_fea
        for block in self.blocks:
            x = block(x)
            distilled_features.append(x)

        trunk = torch.cat(distilled_features, dim=1)
        out_B = self.c1(trunk)
        out_B = self.act(out_B) # SiLU

        out_lr = self.c2(out_B) + out_fea

        output = self.upsampler(out_lr)

        return output