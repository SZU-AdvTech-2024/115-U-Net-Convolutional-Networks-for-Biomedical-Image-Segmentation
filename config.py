import torch
import torchvision.transforms as transforms

class Config:
    def __init__(self):
        self.DATA_DIR = "./data/COVID19" # 数据路径
        # self.DATA_DIR = "./data/ACDC"
        self.BATCH_SIZE = 4 # 批大小
        self.EPOCHS = 1 # 训练轮数
        self.LEARNING_RATE = 0.1 # 学习率
        self.NUM_WORKERS = 4 # 加载数据的线程数
        self.BEST_LOSS = 1e10 # 最佳损失
        self.CHECKPOINTS_DIR = './checkpoints' # 模型保存路径
        self.LOG_DIR = './logs' # 日志保存路径
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 设备
        self.IMG_SIZE = 256 # 图片大小
        self.IN_CHANNELS = 1 # 输入通道数
        self.CLASSES = 4 # 分割类别

        self.transform_image = transforms.Compose([
            transforms.Resize((self.IMG_SIZE, self.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.transform_label = transforms.Compose([
            transforms.Resize((self.IMG_SIZE, self.IMG_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.PILToTensor()
        ])
