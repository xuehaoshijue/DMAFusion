import os 
from PIL import Image
#import cv2
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import numpy as np
import math
from option import opt
from natsort import ns, natsorted

class MyPolDataset(Dataset):
    #构造函数定义
    def __init__(self, path_dir, transform=None,train=False):
        self.train=train
        self.path_dir = path_dir
        self.transform = transform
        self.images_s0 = os.listdir(os.path.join(self.path_dir,'S0'))
        #将path_dir与子目录s0、dolp拼接在一起，os.listdir获得子目录中的图像文件名列表
        self.images_dolp = os.listdir(os.path.join(self.path_dir,'DoLP'))

        self.images_s0 = natsorted(self.images_s0, alg=ns.PATH)
        self.images_dolp = natsorted(self.images_dolp, alg=ns.PATH)

    # 整个数据集有多长
    def __len__(self):
        return len(self.images_s0)

    #获取数据集样本
    def __getitem__(self, index):
        #构建两个图像文件路径，os.path.join将self.path_dir作为基本路径
        # 加s0和dolp文件，再加索引位置图像名
        image_s0_path = os.path.join(self.path_dir,'S0',self.images_s0[index])
        image_dolp_path = os.path.join(self.path_dir,'DoLP',self.images_dolp[index])

        image_s0 = Image.open(image_s0_path).resize((480,640))
        image_dolp = Image.open(image_dolp_path).resize((480,640))

        if self.transform is not None:
          Y_s0= self.transform(image_s0)
          Y_dolp = self.transform(image_dolp)

        return rgb2ycrcb(Y_s0,self.train), rgb2ycrcb(Y_dolp,self.train)

def rgb2ycrcb( img_rgb,train):
    # torch.unsqueeze表示在指定的维度（dim)上增加一个维度
    R = torch.unsqueeze(img_rgb[ 0, :, :], dim=0)
    G = torch.unsqueeze(img_rgb[ 1, :, :], dim=0)
    B = torch.unsqueeze(img_rgb[ 2, :, :], dim=0)
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128 / 255 / 255
    Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128 / 255 / 255
    img_ycbcr = torch.cat([Y, Cr, Cb], 0)

    if train:
        return Y
    else:
        return img_ycbcr
    #return Y

def gradient(input):
    #kernel_x表示x方向上的梯度的卷积核，矩阵，检测图像的边缘、梯度信息
    #unsqueeze(0):表示在最外层(0)添加一个维度，两次扩展维度，适应张量，最后移动到gpu

    kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
    kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).to(opt.device)
    kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
    kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).to(opt.device)
#将input与kernel_x和kernel_y进行卷积，从而计算出梯度

    grad_x = F.conv2d(input, kernel_x)
    grad_y = F.conv2d(input, kernel_y)
    gradient = torch.abs(grad_x) + torch.abs(grad_y)
#torch.abs函数获取每个分量的绝对值
    return gradient

def ycrcb2rgb(Y,Cr,Cb):
    R = Y + 1.404 * (Cr - 128/255/255)
    G = Y - 0.344136 * (Cb - 128/255/255) - 0.714136 * (Cr - 128/255/255)
    B = Y + 1.772 * (Cb - 128/255/255)
    img_rgb = torch.cat([R, G, B],1)
    return img_rgb

#计算图像的熵
def entropy(image, width=480, height=640):
    g = np.histogram(image,bins = 256,range=(0,256))
    num = np.array(g[0])/(width*height)
#np.histogram计算图像中的像素直方图，分成256个区间并计算每个像素中的像素数量
    #计算每个像素值的频率，并归一化
    result=0.
    #迭代了0-255所以的像素值，如果大于零就用公式计算信息熵
    for i in range(0,256):
        if num[i] > 0 :
            result += (-num[i])*math.log(num[i],2)
    return result

#数据增强操作
def data_augmentation(imagein, mode):
    image = imagein
    if mode == 0:
        # original 保持不变
        return image
    elif mode == 1:
        # flip up and down 上下翻转图像
        image = torch.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree 逆时针旋转90度
        image = torch.rot90(image,1,[2,3])
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = torch.rot90(image,1,[2,3])
        image = torch.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        image = torch.rot90(image,2,[2,3])
    elif mode == 5:
        # rotate 180 degree and flip
        image = torch.rot90(image,2,[2,3])
        image = torch.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        image = torch.rot90(image,3,[2,3])
    elif mode == 7:
        # rotate 270 degree and flip
        image = torch.rot90(image, 3,[2,3])
        image = torch.flipud(image)
    imageout = image

    return imageout

