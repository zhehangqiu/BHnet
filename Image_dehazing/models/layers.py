import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# --------------------------------------------------------------------------------
train_size = (1,3,256,256)
#这个自定义池化层允许在训练过程中根据输入大小自动计算池化核的大小，还提供了一个快速实现选项以提高效率。在前向传播过程中，根据不同的实现方式，进行相应的平均池化操作
# kernel_size 表示池化核的大小。
# base_size 表示用于计算池化核大小的基础大小。
# auto_pad 表示是否进行自动填充。
# fast_imp 表示是否使用更快的非等价实现方法。
class AvgPool2d(nn.Module):
    def __init__(self, kernel_size=None, base_size=None, auto_pad=True, fast_imp=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad

        # 仅用于快速实现
        self.fast_imp = fast_imp
        self.rs = [5,4,3,2,1]  # 用于快速实现的参数
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]

    def extra_repr(self) -> str:
        return 'kernel_size={}, base_size={}, stride={}, fast_imp={}'.format(
            self.kernel_size, self.base_size, self.kernel_size, self.fast_imp
        )

    def forward(self, x):
        # 根据输入的 kernel_size 和 base_size 参数计算实际的 kernel_size
        if self.kernel_size is None and self.base_size:
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2]*self.base_size[0]//train_size[-2]
            self.kernel_size[1] = x.shape[3]*self.base_size[1]//train_size[-1]

            # 仅用于快速实现
            self.max_r1 = max(1, self.rs[0]*x.shape[2]//train_size[-2])
            self.max_r2 = max(1, self.rs[0]*x.shape[3]//train_size[-1])

        # 使用快速实现的方法进行平均池化
        if self.fast_imp:
            h, w = x.shape[2:]
            if self.kernel_size[0]>=h and self.kernel_size[1]>=w:
                out = F.adaptive_avg_pool2d(x,1)  # 使用自适应平均池化
            else:
                r1 = [r for r in self.rs if h%r==0][0]
                r2 = [r for r in self.rs if w%r==0][0]
                r1 = min(self.max_r1, r1)
                r2 = min(self.max_r2, r2)
                s = x[:,:,::r1, ::r2].cumsum(dim=-1).cumsum(dim=-2)
                n, c, h, w = s.shape
                k1, k2 = min(h-1, self.kernel_size[0]//r1), min(w-1, self.kernel_size[1]//r2)
                out = (s[:,:,:-k1,:-k2]-s[:,:,:-k1,k2:]-s[:,:,k1:,:-k2]+s[:,:,k1:,k2:])/(k1*k2)
                out = torch.nn.functional.interpolate(out, scale_factor=(r1,r2))
        else:
            n, c, h, w = x.shape
            s = x.cumsum(dim=-1).cumsum(dim=-2)
            s = torch.nn.functional.pad(s, (1,0,1,0)) # 为方便起见，在周围填充0
            k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])
            s1, s2, s3, s4 = s[:,:,:-k1,:-k2],s[:,:,:-k1,k2:], s[:,:,k1:,:-k2], s[:,:,k1:,k2:]
            out = s4+s1-s2-s3
            out = out / (k1*k2)

        # 如果需要自动填充，则对输出进行自动填充
        if self.auto_pad:
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            pad2d = ((w - _w)//2, (w - _w + 1)//2, (h - _h) // 2, (h - _h + 1) // 2)
            out = torch.nn.functional.pad(out, pad2d, mode='replicate')

        return out
# --------------------------------------------------------------------------------


#BasicConv 类是一个通用的卷积模块，可以方便地定义具有不同组件（如归一化和激活函数）的卷积层
class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        # 如果同时启用了偏置参数和规范化参数，则将偏置参数设置为 False
        if bias and norm:
            bias = False
        # 计算卷积层的填充大小，以便保持输入和输出的大小一致
        padding = kernel_size // 2
        layers = list()
        # 如果启用了转置卷积（反卷积），则调整填充大小以适应反卷积操作
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        # 如果启用了规范化参数，将批量归一化层添加到层列表中
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        # 如果启用了 GELU 激活函数参数，将 GELU 激活函数添加到层列表中
        if relu:
            layers.append(nn.GELU())
        # 创建一个包含所有卷积层及其组件的序列模块 nn.Sequential，并将其保存在 self.main 中
        self.main = nn.Sequential(*layers)
    # forward 方法定义了前向传播的过程，将输入 x 通过 self.main 中的卷积层传递，并返回结果
    def forward(self, x):
        return self.main(x)


#这个模块的目的是通过池化操作（自适应平均池化或自定义池化方式）将输入张量分为两部分，并通过可学习的参数调整两部分的权重。最终输出是两部分的加权和。
#FrequencyAdaptivePooling
class Fap(nn.Module):
    def __init__(self, in_channel, mode) -> None:
        super().__init__()
        # 定义可学习参数 fscale_d 和 fscale_h
        self.fscale_d = nn.Parameter(torch.zeros(in_channel), requires_grad=True)
        self.fscale_h = nn.Parameter(torch.zeros(in_channel), requires_grad=True)
        # 根据 mode 参数选择不同的全局平均池化操作
        if mode[0] == 'train':
            self.ap = nn.AdaptiveAvgPool2d((1,1))
        elif mode[0] == 'test':
            if mode[1] == 'Indoor':
                self.ap = AvgPool2d(base_size=246)
            elif mode[1] == 'Outdoor':
                self.ap = AvgPool2d(base_size=210)
    def forward(self, x):
        # 执行全局平均池化操作
        x_d = self.ap(x)
        # 计算水平方向的高频部分
        x_h = (x - x_d) * (self.fscale_h[None, :, None, None] + 1.)
        # 计算低频部分
        x_d = x_d * self.fscale_d[None, :, None, None]
        # 将低频部分和高频部分相加得到最终输出
        return x_d + x_h
#CHAIR调用
class ResBlock2(nn.Module):
    def __init__(self, in_channel, out_channel, mode, filter=False):
        super(ResBlock2, self).__init__()
        # 第一个卷积层
        self.conv1 = BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True)
        # 第二个卷积层
        self.conv2 = BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        # 是否使用动态滤波器的标志
        self.filter = filter
        # 动态滤波器
        self.dyna = Depth_CA(in_channel) if filter else nn.Identity()
        # 局部自适应池化
        self.localap = Patch_Fap(mode, in_channel // 2, patch_size=2)
        # 全局自适应池化
        self.global_ap = Fap(in_channel // 2, mode)

    def forward(self, x):
        # 第一个卷积层 3×3
        out = self.conv1(x)
        #(MDSF)
        if self.filter:

            out = self.dyna(out)
        # 将输出分成全局和局部两部分（MCSF）
        non_local, local = torch.chunk(out, 2, dim=1)
        # 使用全局自适应池化处理全局部分
        non_local = self.global_ap(non_local)
        # 使用局部自适应池化处理局部部分
        local = self.localap(local)
        # 将处理后的两部分结果沿着通道维度拼接回来
        out = torch.cat((non_local, local), dim=1)
        # 通过第二个卷积层 3×3
        out = self.conv2(out)
        # 将输出与输入相加，实现残差连接
        return out + x

#depth_channel_att
class Depth_CA(nn.Module):
    def __init__(self, dim, kernel=3) -> None:
        super().__init__()
        self.kernel = (1, kernel)
        pad_r = pad_l = kernel // 2
        self.pad = nn.ReflectionPad2d((pad_r, pad_l, 0, 0))
        self.conv = nn.Conv2d(dim, kernel * dim, kernel_size=1, stride=1, bias=False, groups=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.filter_act = nn.Tanh()
        self.filter_bn = nn.BatchNorm2d(kernel * dim)
        self.gamma = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))
        self.modulate = SFconv(dim) if filter else nn.Identity()
    def forward(self, x):
        filter = self.filter_bn(self.conv(self.gap(x)))
        filter = self.filter_act(filter)
        b, c, h, w = filter.shape
        filter = filter.view(b, self.kernel[1], c // self.kernel[1], h * w).permute(0, 1, 3, 2).contiguous()
        B, C, H, W = x.shape
        out = x.permute(0, 2, 3, 1).view(B, H * W, C).unsqueeze(1)
        out = F.unfold(self.pad(out), kernel_size=self.kernel, stride=1)
        out = out.view(B, self.kernel[1], H * W, -1)
        out = torch.sum(out * filter, dim=1, keepdim=True).permute(0, 3, 1, 2).reshape(B, C, H, W)
        out = out * self.gamma + x * self.beta
        out = self.modulate(out)
        return out

#CHAIR模块
class SFconv(nn.Module):
    def __init__(self, features, M=4, r=2, L=32) -> None:
        super().__init__()

        d = max(int(features / r), L)
        self.features = features
        self.convs = nn.ModuleList([])
        self.convh = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        self.convm = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        self.convl = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        self.convll = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        self.fc = nn.Conv2d(features, d, 1, 1, 0)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Conv2d(d, features, 1, 1, 0)
            )
        self.softmax = nn.Softmax(dim=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.out = nn.Conv2d(features, features, 1, 1, 0)
        self.gamma = nn.Parameter(torch.zeros((1, features, 1, 1)), requires_grad=True)

    def forward(self, x):
        ll = self.convll(x)
        l = self.convl(ll)
        m = self.convm(l)
        h = self.convh(m)
        emerge = l + m + h + ll
        emerge = self.gap(emerge)

        fea_z = self.fc(emerge)

        h_a = self.fcs[0](fea_z)
        m_a = self.fcs[1](fea_z)
        l_a = self.fcs[2](fea_z)
        ll_a = self.fcs[3](fea_z)

        attention_vectors = torch.cat([h_a, m_a, l_a, ll_a], dim=1)

        attention_vectors = self.softmax(attention_vectors)
        h_a, m_a, l_a, ll_a = torch.chunk(attention_vectors, 4, dim=1)

        f_h = h * h_a
        f_m = m * m_a
        f_l = l * l_a
        f_ll = ll * ll_a
        out = self.out(f_h + f_m + f_l + f_ll)
        return out * self.gamma + x
#这是一个局部自适应池化模块 Patch_ap，它基于输入的特征图 x，
# 通过将特征图划分为多个小块（patch），然后对每个小块进行自适应平均池化操作
class Patch_Fap(nn.Module):
    def __init__(self, mode, inchannel, patch_size):
        super(Patch_Fap, self).__init__()

        # 根据模式选择全局平均池化层
        if mode[0] == 'train':
            self.ap = nn.AdaptiveAvgPool2d((1,1))
        elif mode[0] == 'test':
            if mode[1] == 'Indoor':
                self.ap = AvgPool2d(base_size=246)
            elif mode[1] == 'Outdoor':
                self.ap = AvgPool2d(base_size=210)
        # 设置 patch 大小和通道数
        self.patch_size = patch_size
        self.channel = inchannel * patch_size ** 2
        # 学习的参数，用于加权计算高频和低频部分
        self.h = nn.Parameter(torch.zeros(self.channel))
        self.l = nn.Parameter(torch.zeros(self.channel))

    def forward(self, x):
        # 将特征图 x 划分为多个小块
        patch_x = rearrange(x, 'b c (p1 w1) (p2 w2) -> b c p1 w1 p2 w2', p1=self.patch_size, p2=self.patch_size)
        patch_x = rearrange(patch_x, ' b c p1 w1 p2 w2 -> b (c p1 p2) w1 w2', p1=self.patch_size, p2=self.patch_size)

        # 对每个小块进行自适应平均池化
        low = self.ap(patch_x)
        # 计算高频部分
        high = (patch_x - low) * self.h[None, :, None, None]
        # 计算最终输出，加权计算高频和低频部分
        out = high + low * self.l[None, :, None, None]
        # 将输出重新组织为原来的形状
        out = rearrange(out, 'b (c p1 p2) w1 w2 -> b c (p1 w1) (p2 w2)', p1=self.patch_size, p2=self.patch_size)
        return out

