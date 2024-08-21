import torch,os,sys,torchvision,argparse
import torch,warnings
#禁止警告消息输出
warnings.filterwarnings('ignore')
#创建解析命令行参数
parser = argparse.ArgumentParser()
#指定在哪个设备上运行代码，paser.add_argument定义命令行参数的
parser.add_argument('--device', type=str, default='cpu')
#定义bs,整数型，默认值4，批量大小
parser.add_argument('--bs', type=int, default=4, help='batch size')
#定义lr浮点数类型，默认0.01，设置学习率
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
#定义学习率衰减率，浮点数，默认0.99
parser.add_argument('--lr_decay_rate', type=float, default=0.99, help='lr decay rate')
#定义训练轮数，整数类型，默认100
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--input_channel', type=int, default=1)
parser.add_argument('--output_channel', type=int, default=1)
#用于指定训练数据集目录，字符串型
parser.add_argument('--traindata_dir', type=str, default='./dataset/train')
#指定测试集数据集目录，字符串型
parser.add_argument('--testdata_dir', type=str, default='./dataset/after')
#表示测试模型的目录和文件名
parser.add_argument('--testmodel_dir', type=str, default="./models/model_50.pth")
#是否要从检查点中恢复模型训练
parser.add_argument('--resume', type=bool,default=False)

#将结果存储在opt中
opt=parser.parse_args()
#检查代码系统中是否有cuda，否则设置为cpu,用来选择设备
opt.device='cuda' if torch.cuda.is_available() else 'cpu'

print(opt)
#检查是否有model,logs,output目录，没有就创建
if not os.path.exists('models'):
	os.mkdir('models')
if not os.path.exists('logs'):
	os.mkdir('logs')
if not os.path.exists('output'):
	os.mkdir('output')

