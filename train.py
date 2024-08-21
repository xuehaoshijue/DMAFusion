import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设置gpu
import utils
import torch
#import cv2
import torchvision
import numpy as np
from pytorch_msssim import msssim
from pytorch_msssim import ssim
from torch.optim import Adam  # 优化器
from net import FusionNet
from tqdm import tqdm
from torch.utils.data import DataLoader  # 加载数据集
from tensorboardX import SummaryWriter
from option import opt
import torch.nn.functional as F

#import matplotlib.pyplot as plt
from collections import OrderedDict

from typing import Any ,Callable,List,Optional,Sequence,Tuple

def train():
    ## prepare data#torchvision.transforms.Compose将预处理操作组合在一起，将图像数据转为张量
    data_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    train_dataset = utils.MyPolDataset(
        opt.traindata_dir, transform=data_transform, train=True
    )
    # 创建数据加载器，按照原顺序加载，不对数据随机重排
    train_loader = DataLoader(dataset=train_dataset, batch_size=opt.bs, shuffle=False)
    ## net

    densefuse_model = FusionNet()(opt.input_channel, opt.output_channel)
    # 调用initialize_weights初始化权重参数
    densefuse_model.initialize_weights()
    # 将模型移动到指定设备
    densefuse_model = densefuse_model.to(device=opt.device)
    ## Optimizer 优化器 更新参数，学习率
    # 学习率调度器，每个训练周期后，学习率乘以gamma
    optimizer = Adam(densefuse_model.parameters(), opt.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=opt.lr_decay_rate
    )
    #SSIM_fun = ssim
    ## log
    writer = SummaryWriter("logs/")
    print("Start training.....")
    for epoch in range(opt.epochs):
        loss = 0.0
        batch_train = 0.0
        densefuse_model.train()
        with tqdm(train_loader, unit="batch") as tepoch:  ##set tqdm创建进度条
            for Y_s0, Y_dolp in tepoch:
                optimizer.zero_grad()  # 每个批次开始优化器的梯度缓冲区清零
                mode = np.random.permutation(8)  # 随机生成长度为8的排列数组
                Y_s0 = Y_s0.to(device=opt.device)
                Y_dolp = Y_dolp.to(device=opt.device)
                # print(mode)
                # print(Y_s0.shape, Y_dolp.shape)
                Y_s0 = utils.data_augmentation(Y_s0, mode[0])  # 数据增强操作，随机生成的mode
                Y_dolp = utils.data_augmentation(Y_dolp, mode[0])
                # print(Y_s0.shape, Y_dolp.shape)
                # return
                outputs = densefuse_model(Y_s0, Y_dolp)  # 增强后传给densefuse_model

                ## ssim_loss ，使用ssim函数来计算ssimloss，true表示进行标准化
                ssim_loss_value = (ssim(Y_s0, outputs, size_average=True).mean() + ssim(Y_dolp, outputs,
                                                 size_average=True).mean()) / 2
                #ssim_loss_value = msssim(Y_s0, Y_dolp, outputs, normalize=True)
                ## intensity_loss 权重w1调整强度损失
                # 计算强度损失
                #w1 = torch.exp(Y_s0 / 0.1) / (
                    #torch.exp(Y_s0 / 0.1) + torch.exp(Y_dolp / 0.1))
                #intensity_loss = torch.mean(w1 * ((Y_s0 - outputs) ** 2)) + torch.mean(
                    #(1 - w1) * ((Y_dolp - outputs) ** 2))
                b = torch.max(Y_s0,Y_dolp )
                intensity_loss = F.l1_loss(b,outputs)

                a = torch.max(torch.abs(utils.gradient(Y_s0)), torch.abs(utils.gradient(Y_dolp)))
                grad_loss = F.l1_loss(torch.abs(utils.gradient(outputs)),a)
                ## total loss
                total_loss = 6 * (1-ssim_loss_value )+ 16 * intensity_loss + 46 * grad_loss

                #if len(ssim_loss_value.shape) > 0:
                    #print("This variable is a tensor.")
                #else:
                    #print("This variable is a scalar.")

                total_loss.backward()  # 反向传播计算梯度，
                # print(densefuse_model.conv6.conv2d.weight.shape)
                # print(densefuse_model.conv6.conv2d.weight[0].grad)
                optimizer.step()  # 更新模型的权重参数
                loss = loss + total_loss
                batch_train = batch_train + 1
                # 用来计算每轮结束的平均损失
                tepoch.set_description_str(
                    "Epoch: {:d},loss: {:f},total_loss: {:f}".format(
                        epoch + 1, total_loss, loss / batch_train
                    )
                )
            # 显示当前轮次的损失信息
        # 512个批次进行一次调整，记录损失函数和全局步数
        scheduler.step()
        writer.add_scalar("train_loss", loss / batch_train, global_step=epoch + 1)

        ## save model
        densefuse_model.eval()  # 切换到评估模式
        save_model_path = "./models/model_" + str(epoch + 1) + ".pth"  # 保存模型路径
        torch.save(densefuse_model.state_dict(), save_model_path)

    writer.close()
    print("\nDone, trained model saved at", save_model_path)
if __name__ == "__main__":  # 检查脚本是否以主程序的形式运行
    train()


