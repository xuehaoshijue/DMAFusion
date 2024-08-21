import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import utils
import torch
import torchvision
#import torchvision.transforms.functional as TF
from net import DenseFuse_net
from torch.utils.data import DataLoader
#from PIL import Image
# from imageio import imsave
import time
from option import opt
#import cv2

def test():
    ##prepare data
    # 将图像数据转为张量格式
    data_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    # opt.testdata_dir测试数据的路径，将data_transform应用于每个样本
    train_dataset = utils.MyPolDataset(
        opt.testdata_dir, transform=data_transform, train=False
    )
    # 数据加载器，shuffle表示不对数据进行随机重排
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)

    ##load net
    with torch.no_grad():  # 不计算梯度
        print("testing")
        test_model = DenseFuse_net(opt.input_channel, opt.output_channel)
        # 加载之前训练好的模型权重参数，opt.testmodel_dir模型的保存路径
        test_model.load_state_dict(torch.load(opt.testmodel_dir))
        test_model = test_model.to(device=opt.device)
        # 模型设置为评估模式，不会进行梯度的计算和训练
        test_model.eval()
        begin_time = time.time()  # 记录当前时间
        # 开始循环，images_s0,images_dolp是一对输入数据
        for i, (images_s0, images_dolp) in enumerate(train_loader):
            images_s0 = images_s0.to(device=opt.device)
            images_dolp = images_dolp.to(device=opt.device)
            # img_fusion = test_model(images_s0, images_dolp)l
            img_fusion = test_model(
                torch.unsqueeze(images_s0[:, 0, :, :], dim=1),
                torch.unsqueeze(images_dolp[:, 0, :, :], dim=1),
            )
            torchvision.utils.save_image(
                utils.ycrcb2rgb(
                    img_fusion,
                    torch.unsqueeze(images_s0[:, 1, :, :], dim=1),
                    torch.unsqueeze(images_s0[:, 2, :, :], dim=1),
                ),
                "./output/" + str(i) + ".png",
            )
    proc_time = time.time() - begin_time
    print("Total processing time: {:.3}s".format(proc_time))

if __name__ == "__main__":
    test()
