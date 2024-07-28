import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *
#append是添加的到末尾
# Encoder Block
#整个块的作用是对输入 x 进行多次残差块的特征提取
class EBlock(nn.Module):
    def __init__(self, out_channel, num_res, mode):
        super(EBlock, self).__init__()

        layers = [ResBlock2(out_channel, out_channel, mode) for _ in range(num_res-1)]
        layers.append(ResBlock2(out_channel, out_channel, mode, filter=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# Decoder Block
#与 EBlock 类似，这个块的作用是对输入 x 进行多次残差块的特征提取
class DBlock(nn.Module):
    def __init__(self, channel, num_res, mode):
        super(DBlock, self).__init__()

        layers = [ResBlock2(channel, channel, mode) for _ in range(num_res-1)]
        layers.append(ResBlock2(channel, channel, mode, filter=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

#SCM 是一个空间上下文模块，通过一系列的卷积和实例归一化操作来提取输入的空间信息
class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        # 创建一个包含所有卷积层及其组件的序列模块 nn.Sequential，并将其保存在 self.main 中
        self.main = nn.Sequential(
            # 第一层卷积层，输入通道数为 3，输出通道数为 out_plane 的四分之一，使用 3x3 的卷积核，步幅为 1，启用 ReLU 激活函数
            BasicConv(3, out_plane // 4, kernel_size=3, stride=1, relu=True),
            # 第二层卷积层，输入通道数为 out_plane 的四分之一，输出通道数为 out_plane 的二分之一，使用 1x1 的卷积核，步幅为 1，启用 ReLU 激活函数
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            # 第三层卷积层，输入通道数为 out_plane 的二分之一，输出通道数为 out_plane 的二分之一，使用 3x3 的卷积核，步幅为 1，启用 ReLU 激活函数
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            # 第四层卷积层，输入通道数为 out_plane 的二分之一，输出通道数为 out_plane，使用 1x1 的卷积核，步幅为 1，禁用 ReLU 激活函数
            BasicConv(out_plane // 2, out_plane, kernel_size=1, stride=1, relu=False),
            # 对输出通道数为 out_plane 的特征图进行实例归一化
            nn.InstanceNorm2d(out_plane, affine=True)
        )
    def forward(self, x):
        x = self.main(x)
        return x

#FAM 接收两个特征张量 x1 和 x2，并通过堆叠这两个特征的通道维度，然后通过卷积层进行融合。
class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel*2, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        return self.merge(torch.cat([x1, x2], dim=1))




class BHNet(nn.Module):
    def __init__(self, mode, num_res=16):
        super(BHNet, self).__init__()

        base_channel = 32

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res, mode),
            EBlock(base_channel*2, num_res, mode),
            EBlock(base_channel*4, num_res, mode),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res, mode),
            DBlock(base_channel * 2, num_res, mode),
            DBlock(base_channel, num_res, mode)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)



    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5) #对输入进行尺度缩小
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()
        research = list()
        # 256*256
        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)
        # 128*128
        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)
        # 64*64
        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)


        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        # 128*128
        z = self.feat_extract[3](z)
        outputs.append(z_+x_4)
        #research.append(z_)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)

        # 256*256
        z = self.feat_extract[4](z)
        outputs.append(z_+x_2)
        #research.append(z_)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs.append(z+x)
        #research.append(z)


        return outputs



def build_net(mode):
    return BHNet(mode)
