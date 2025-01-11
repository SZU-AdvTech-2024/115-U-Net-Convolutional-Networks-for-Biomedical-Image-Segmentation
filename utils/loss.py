import torch
import torch.nn as nn
from monai.losses import DiceLoss
import os

class ComboLoss(nn.Module):
    def __init__(self, dataset_name, alpha=0.5, smooth=1e-5):
        super(ComboLoss, self).__init__()
        self.dice_loss = DiceLoss(include_background=True, to_onehot_y=True, softmax=True, smooth_nr=smooth, smooth_dr=smooth)
        self.ce_loss = nn.CrossEntropyLoss()
        self.alpha = alpha
        self.dataset_name = dataset_name

    def convert_labels(self, labels):
        new_labels = torch.zeros_like(labels)
        if self.dataset_name == "ACDC":
            new_labels[labels == 85] = 1   # RV
            new_labels[labels == 170] = 2  # MYO
            new_labels[labels == 255] = 3  # LV
        elif self.dataset_name == "COVID19":
            new_labels[labels == 255] = 1  # Tumor
        else:
            raise ValueError(f"Unsupported dataset name: {self.dataset_name}")
        return new_labels

    def forward(self, preds, labels):
        labels = self.convert_labels(labels)

        # 确保标签值在有效范围内
        num_classes = preds.shape[1]
        labels = torch.clamp(labels, min=0, max=num_classes - 1)

        # 确保标签数据类型为 torch.long
        labels = labels.long()

        # 计算Dice Loss
        dice = self.dice_loss(preds, labels)

        # 计算Cross Entropy Loss
        ce = self.ce_loss(preds, labels.squeeze(1))

        # 加权求和
        return self.alpha * dice + (1 - self.alpha) * ce
