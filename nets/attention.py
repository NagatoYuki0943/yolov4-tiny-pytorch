import torch
import torch.nn as nn
import math


#---------------------------------------------------#
#   se,通道注意力
#   特征层高宽全局平均池化,进行两次全连接层,第一次降低特征数,第二次还原特征数,进行sigmoid变换到0~1之间
#   最后将原值和输出结果相乘
#---------------------------------------------------#
class se_block(nn.Module):
    def __init__(self, channel, ratio=16):
        super(se_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // ratio, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channel // ratio, channel, bias=False),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

#---------------------------------------------------#
#   cbam通道注意力
#   将输入内容在宽高上分别进行平均池化和最大池化,然后经过共用的两个全连接层,然后将两个结果相加,取sigmoid,最后和原值相乘
#---------------------------------------------------#
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 将输入内容分别进行平均池化和最大池化,然后经过共用的两个全连接层
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # 将两个结果相加,取sigmoid
        out = avg_out + max_out
        return self.sigmoid(out)

#---------------------------------------------------#
#   cbam空间注意力
#   不关注通道数量,所以不用给channel
#   在每一个特征点的通道上取最大值和平均值。
#   之后将这两个结果进行一个堆叠，利用一次输出通道数为1的卷积调整通道数，然后取一个sigmoid
#   在获得这个权值后，我们将这个权值乘上原输入特征层即可。
#   b, c, h, w -> b, 1, h, w
#---------------------------------------------------#
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # b, c, h, w -> b, 1, h, w
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # b, c, h, w -> b, 1, h, w
        max_out, _ = torch.max(x, dim=1, keepdim=True)      # 注意 torch.max(x, dim, keepdim=True) 返回值和下标
        # b, 1, h, w + b, 1, h, w -> b, 2, h, w
        x = torch.cat([avg_out, max_out], dim=1)
        # b, 2, h, w -> b, 1, h, w
        x = self.conv1(x)
        return self.sigmoid(x)

#---------------------------------------------------#
#   cbam: 通道+空间注意力
#   先进行通道,再进行宽高
#---------------------------------------------------#
class cbam_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x*self.channelattention(x)
        x = x*self.spatialattention(x)
        return x


#---------------------------------------------------#
#   eca
#   去除原来SE模块中的全连接层，直接在全局平均池化之后的特征上通过一个1D卷积进行学习
#   全连接层使用上一层全部数据得到下一层的全部数据
#   1D卷积使用上一层的n个数据得到下一层的全部数据,减少计算量
#---------------------------------------------------#
class eca_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(eca_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 1d卷积,特征长条上进行特征提取
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        # b, c, 1, 1 -> b, c, 1 -> b, 1, c 进行1D卷积 对C进行计算,相当于得到每个通道的权重
        # b, 1, c -> b, c, 1 -> b, c, 1, 1
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

#---------------------------------------------------#
#   ca
#---------------------------------------------------#
class CA_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA_Block, self).__init__()

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel//reduction, kernel_size=1, stride=1, bias=False)

        self.relu   = nn.ReLU()
        self.bn     = nn.BatchNorm2d(channel//reduction)

        self.F_h = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.F_w = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        # batch_size, c, h, w
        _, _, h, w = x.size()

        # batch_size, c, h, w => batch_size, c, h, 1 => batch_size, c, 1, h
        x_h = torch.mean(x, dim = 3, keepdim = True).permute(0, 1, 3, 2)
        # batch_size, c, h, w => batch_size, c, 1, w
        x_w = torch.mean(x, dim = 2, keepdim = True)

        # batch_size, c, 1, w cat batch_size, c, 1, h => batch_size, c, 1, w + h
        # batch_size, c, 1, w + h => batch_size, c / r, 1, w + h
        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))

        # batch_size, c / r, 1, w + h => batch_size, c / r, 1, h and batch_size, c / r, 1, w
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        # batch_size, c / r, 1, h => batch_size, c / r, h, 1 => batch_size, c, h, 1
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        # batch_size, c / r, 1, w => batch_size, c, 1, w
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out
