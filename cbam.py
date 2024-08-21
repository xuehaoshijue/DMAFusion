import torch
import math
import torch.nn as nn
import torch.nn.functional as F

#定义类basicconv
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        #输入通道数，输出通道数，卷积核大小，不长，填充，膨胀率，组数，是否使用relu激活函数，是否归一化，是否添加偏置
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None
#归一化层，eps归一化的小正数，防止分母为零，momentum用于调整均值和方差，是否应用缩放和偏置
    #前向传播，处理x
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
#对张量进行压扁操作，x.size(0)返回张量x的第一个维度大小，通常是batch.size
#将张量x从多维压扁成一个一维形状，第一个维度不变，其他维度展开到一维中
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

#执行通道的门控操作：输入张量的通道数，通道门控的压缩比例通常为16，池化类型裂变
class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        #定义了一个多层感知机，计算通道注意力
        self.mlp = nn.Sequential(
            Flatten(),#将输入张量转化为一维
            #linear是全链接层类，一个是输入大小，一个是输出大小
            nn.Linear(gate_channels, gate_channels // reduction_ratio),#创建一个线性层将输入特征维度减小到gate_channels // reduction_ratio
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)#将维度恢复，//是整数除法运算符
            )
        self.pool_types = pool_types#将池化类型存储在模块中

    def forward(self, x):
        channel_att_sum = None     #初始化变量
        for pool_type in self.pool_types:  #循环遍历列表中的每种池化方式
            if pool_type=='avg':
                #对x进行平均池化，x.size(2)定义了高度，x.size定义宽度
                #将池化后的结果传递给self.mlp，用来计算通道注意力分数
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':  #2表示用l2范数来计算池化，
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only logsumexp_2d(x) = log(sum(exp(x)))
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
#sigmoid激活函数表达式 1/1+exp(-x)，在2和3维上分别多加一个维度，将结果扩展到与x相同的形状
        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    #-1自动计算第三个维度，每个通道内的值放在单独的一个维度中
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    #计算每个通道的最大值，并储存在s中，在dim = 2上运行，是否保持操作后的维度
    #找到空间维度的最大值
    #减去通道内的最大值，避免梯度爆炸

    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

#(x,1)指在张量x的第一个维度，开始操作
#找到每个通道内的最大值，返回在[0]中，然后添加一个维度，mean计算平均值
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
#torch.cat将其的dim = 1维度上连接起来，实现通道池化

#空间注意力
class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7   #卷积核大小为7
        self.compress = ChannelPool()
        #卷积操作 输入通道2，输出通道1，填充，不应用relu激活函数
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)  #将x_compress传递给spatial
        scale = F.sigmoid(x_out) # broadcasting权重张量
        return x * scale

#实现通道注意和空间注意力
#输入通道，降维比例，是否使用空间布控
class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        #gate_channels表示输入通道数
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()#空间注意力
    def forward(self, x):
        if not self.no_spatial:
            x_out = self.SpatialGate(x)
        x_out = self.ChannelGate(x_out)#通道注意力
        return x_out


