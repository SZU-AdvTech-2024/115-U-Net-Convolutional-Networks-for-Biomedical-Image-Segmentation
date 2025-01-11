import torch
import torch.nn as nn
import torch.optim as optim
from config import Config
from utils.dataset import Transform, Convid19Dataset, ACDCDataset
from torch.utils.data import DataLoader, random_split
from unet.unet import UNet
from utils.metrics import dice_coefficient, iou_score, pixel_error
from utils.loss import ComboLoss
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def train(config):
    # 加载数据集
    transform = Transform(config.transform_image, config.transform_label)
    dataset_name = os.path.basename(config.DATA_DIR)
    match dataset_name:
        case "COVID19":
            dataset = Convid19Dataset(root_dir=config.DATA_DIR, transform=transform)
            train_size = int(len(dataset)*0.8)
            test_size = len(dataset) - train_size
            torch.manual_seed(0)
            train_set, test_set = random_split(dataset, [train_size, test_size])
        case "ACDC":
            train_set = ACDCDataset(root_dir=config.DATA_DIR, phase="training", transform=transform)
            test_set = ACDCDataset(root_dir=config.DATA_DIR, phase="testing", transform=transform)
        
    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=config.BATCH_SIZE, shuffle=True,
                             num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True)
    
    # 定义模型
    unet = UNet(config.IN_CHANNELS, config.IN_CHANNELS * config.CLASSES).to(config.DEVICE)

    # 定义优化器
    optimizer = optim.Adam(unet.parameters(), lr=config.LEARNING_RATE)

    # 定义损失函数
    alpha = 0.5
    criterion = ComboLoss(dataset_name=dataset_name, alpha=alpha)

    # 添加学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 日志记录
    writer = SummaryWriter(os.path.join(config.LOG_DIR, dataset_name, "runs"))
    checkpoints_dir = os.path.join(config.CHECKPOINTS_DIR, dataset_name)
    os.makedirs(checkpoints_dir, exist_ok=True)

    # 打印相关信息
    print(f"Using dataset {dataset_name}")
    print(f"Train on {len(train_set)} samples, test on {len(test_set)} samples")
    print(f"Using device {config.DEVICE}")

    for epoch in range(config.EPOCHS):
        # 训练
        unet.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1:3}/{config.EPOCHS}[train]", ncols=100):
            image = batch['image'].to(config.DEVICE, dtype=torch.float32)
            label = batch['label'].to(config.DEVICE, dtype=torch.float32)

            optimizer.zero_grad()
            pred = unet(image)
            loss = criterion(pred, label)
            train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
            optimizer.step()
            
        # 测试
        unet.eval()
        test_loss = 0
        dice_total = 0
        iou_total = 0
        pixel_err_total = 0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Epoch {epoch+1:3}/{config.EPOCHS}[test ]", ncols=100):
                image = batch['image'].to(config.DEVICE, dtype=torch.float32)
                label = batch['label'].to(config.DEVICE, dtype=torch.float32)
                pred = unet(image)
                loss = criterion(pred, label)
                test_loss += loss.item()

                # 计算metrics
                pred = torch.argmax(pred, dim=1).cpu().numpy() # [B, H, W]
                label = torch.squeeze(label, dim=1).cpu().numpy() # [B, H, W]
                iou_total += iou_score(pred, label, config.CLASSES, dataset_name)
                dice_total += dice_coefficient(pred, label, config.CLASSES, dataset_name)
                pixel_err_total += pixel_error(pred, label, dataset_name)


                
        # 计算平均损失值和metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)
        avg_dice = dice_total / len(test_loader)
        avg_iou = iou_total / len(test_loader)
        avg_pixel_err = pixel_err_total / len(test_loader)

        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Test  Loss: {avg_test_loss:.4f}")
        print(f"Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f}, Pixel Error: {avg_pixel_err:.4f}")

        # 更新学习率
        scheduler.step(avg_test_loss)

        # # 保存最好的模型权重
        if avg_test_loss < config.BEST_LOSS:
            config.BEST_LOSS = avg_test_loss
            torch.save(unet.state_dict(), os.path.join(checkpoints_dir, 'best.pth'))
            print(f"Best model saved at epoch {epoch+1}!")

        # 保存最后一个模型权重
        if epoch == config.EPOCHS - 1:
            torch.save(unet.state_dict(), os.path.join(checkpoints_dir, 'last.pth'))
            print(f"Last model saved at epoch {epoch+1}!")
        
        # 将损失写入日志文件
        writer.add_scalars("Loss", {"train": avg_train_loss, "test": avg_test_loss}, epoch+1)
        writer.add_scalars("Dice", {"test": avg_dice}, epoch+1)
        writer.add_scalars("IoU", {"test": avg_iou}, epoch+1)
        writer.add_scalars("Pixel Error", {"test": avg_pixel_err}, epoch+1)

    writer.close()
    

if __name__ == "__main__":
    config = Config()
    train(config)

    
