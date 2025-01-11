import torch
import torchvision.transforms as transforms
from PIL import Image
from unet.unet import UNet
import matplotlib.pyplot as plt
import numpy as np
from config import Config
import argparse
from args import add_predict_args
import os

def visualize_all_segmentations(images, preds):
    """ 在同一画布中显示所有预测结果 """
    num_images = len(images)
    fig, axes = plt.subplots(num_images, 3, figsize=(4, 2 * num_images))

    if num_images == 1:
        axes = [axes]

    for i, (input_np, pred_np) in enumerate(zip(images, preds)):
        # 显示输入图像
        if i == 0:
            axes[i][0].set_title('Image')
        axes[i][0].imshow(input_np, cmap='gray')
        axes[i][0].axis('off')

        # 显示掩膜
        if i == 0:
            axes[i][1].set_title('Mask')
        axes[i][1].imshow(pred_np, cmap='gray')
        axes[i][1].axis('off')

        # 显示分割结果
        if i == 0:
            axes[i][2].set_title('Segmentation')
        overlay = input_np.copy()
        overlay[pred_np == 1] = [255, 0, 0]
        overlay[pred_np == 2] = [0, 255, 0]
        overlay[pred_np == 3] = [0, 0, 255]
        axes[i][2].imshow(overlay.astype(np.uint8), cmap='gray')
        axes[i][2].axis('off')

    # 自动调整子图间距
    plt.tight_layout(h_pad=0.5, w_pad=0.2)
    plt.subplots_adjust(top=0.95)
    plt.show()

def predict(config, args):
    # 创建UNet模型实例
    unet = UNet(config.IN_CHANNELS, config.IN_CHANNELS * config.CLASSES).to(config.DEVICE)

    # 加载模型权重
    assert os.path.exists(args.weights_path), "Weights path does not exist."
    checkpoint = torch.load(args.weights_path, map_location=config.DEVICE)
    unet.load_state_dict(checkpoint)
    unet.eval()

    # 检查输入文件夹
    assert os.path.exists(args.image_dir), "Image directory does not exist."
    image_files = [f for f in os.listdir(args.image_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    assert len(image_files) > 0, "No image files found in the directory."

    all_images = []
    all_preds = []

    for image_file in image_files:
        image_path = os.path.join(args.image_dir, image_file)
        img = Image.open(image_path).convert("L")
        img_tensor = config.transform_image(img).unsqueeze(0)
        img_tensor = img_tensor.to(config.DEVICE, dtype=torch.float32)

        img = Image.open(image_path).convert("RGB")
        img = img.resize((config.IMG_SIZE, config.IMG_SIZE))
        img_np = np.array(img)

        with torch.no_grad():
            pred = unet(img_tensor)

        pred_np = torch.argmax(pred, dim=1).cpu().numpy()
        pred_np = pred_np.squeeze()

        all_images.append(img_np)
        all_preds.append(pred_np)

    # 在同一画布中可视化所有结果
    visualize_all_segmentations(all_images, all_preds)

if __name__ == "__main__":
    config = Config()
    parser = argparse.ArgumentParser("Predicting", allow_abbrev=False)
    add_predict_args(parser)
    args = parser.parse_args()
    predict(config, args)
