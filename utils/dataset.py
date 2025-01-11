import os
import glob
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


class Transform:
    def __init__(self, transform_image=None, transform_label=None):
        self.transform_image = transform_image
        self.transform_label = transform_label

    def __call__(self, image, label=None):
        if self.transform_image:
            image = self.transform_image(image)
        if self.transform_label:
            label = self.transform_label(label)
        return image, label


# 定义数据集Covid19Dataset
class Convid19Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): 数据集的根目录路径。
            transform (callable, optional): 可选的图像和标签变换。
        """
        self.transform = transform
        self.images_path = sorted(glob.glob(os.path.join(root_dir, 'images', '*.bmp')))
        self.labels_path = sorted(glob.glob(os.path.join(root_dir, 'labels', '*.bmp')))
        assert len(self.images_path) == len(self.labels_path), "Number of images and labels do not match."

    def __len__(self):
        return len(self.images_path)
    

    def __getitem__(self, idx):
        image_path = self.images_path[idx]
        label_path = self.labels_path[idx]
        image = Image.open(image_path).convert('L')
        label = Image.open(label_path).convert('L')
        
        if self.transform:
            image, label = self.transform(image, label)
        else:
            image_np = np.array(image)
            label_np = np.array(label)
            
            # 归一化处理，防止除零
            image_min, image_max = np.min(image_np), np.max(image_np)
            if image_max > image_min:  # 避免除零
                image_np = (image_np - image_min) / (image_max - image_min)
            else:
                image_np = np.zeros_like(image_np)  # 若图像全为常量，直接置零
            
            # 转换为 PyTorch 张量
            image = torch.tensor(image_np).unsqueeze(0).float()
            label = torch.tensor(label_np).unsqueeze(0).long()
    
        return {"image": image, "label": label}



# 定义数据集ACDCDataset
class ACDCDataset(Dataset):
    """
    自定义数据集类，用于读取 ACDC 数据集中的 frameXX 和其 _gt 标签。
    """
    def __init__(self, root_dir, phase='training', transform=None):
        """
        Args:
            root_dir (str): 数据集的根目录路径。
            phase (str): 'training' 或 'testing'，选择训练或测试集。
            transform (callable, optional): 图像和标签的可选变换。
        """
        self.transform = transform
        self.images_path = []
        self.labels_path = []
        phase_path = os.path.join(root_dir, phase)
        patients_path = [entry.path for entry in os.scandir(phase_path) if entry.is_dir()]
        patients_path.sort()

        for patient_path in patients_path:
            for entry in os.scandir(patient_path):
                if not entry.is_dir():
                    continue
                if "gt" in entry.name:
                    for file in os.listdir(entry.path):
                        self.labels_path.append(os.path.join(entry.path, file))
                elif "frame" in entry.name:
                    for file in os.listdir(entry.path):
                        self.images_path.append(os.path.join(entry.path, file))

        self.images_path.sort()
        self.labels_path.sort()
        
        # 确保图像和标签数量一致
        assert len(self.images_path) == len(self.labels_path), "Number of images and labels do not match."


    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        image_path = self.images_path[idx]
        label_path = self.labels_path[idx]
        image = Image.open(image_path).convert('L')
        label = Image.open(label_path).convert('L')
        
        if self.transform:
            image, label = self.transform(image, label)
        else:
            image_np = np.array(image)
            label_np = np.array(label)
            
            # 归一化处理，防止除零
            image_min, image_max = np.min(image_np), np.max(image_np)
            if image_max > image_min:  # 避免除零
                image_np = (image_np - image_min) / (image_max - image_min)
            else:
                image_np = np.zeros_like(image_np)  # 若图像全为常量，直接置零
            
            # 转换为 PyTorch 张量
            image = torch.tensor(image_np).unsqueeze(0).float()
            label = torch.tensor(label_np).unsqueeze(0).long()
    
        return {"image": image, "label": label}



# 用法示例
if __name__ == "__main__":

    transform_image = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5),
    ])
    transform_label = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor()
    ])
    transform = Transform(transform_image, transform_label)

    root_dir = "data\\ACDC"
    dataset = ACDCDataset(root_dir=root_dir, phase='testing', transform=transform)
    # root_dir = "data\\COVID19"
    # dataset = Convid19Dataset(root_dir=root_dir, transform=transform)

    # 测试数据集
    print("训练集样本数量:", len(dataset))
    image, label = dataset[0]["image"], dataset[0]["label"]
    print("图像大小:", image.shape)
    print(image)
    print("标签大小:", label.shape)
    print(label)

