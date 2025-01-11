import numpy as np

def convert_labels(labels, dataset_name):
    if not isinstance(labels, np.ndarray):
        raise ValueError("Input labels must be a numpy array.")
    new_labels = np.zeros_like(labels)
    if dataset_name == "ACDC":
        new_labels[labels ==  85] = 1  # RV
        new_labels[labels == 170] = 2  # MYO
        new_labels[labels == 255] = 3  # LV
    elif dataset_name == "COVID19":
        new_labels[labels == 255] = 1  # Tumor
    else:
        raise ValueError("Unsupported dataset_name provided.")
    return new_labels


def dice_coefficient(y_pred, y_true, num_classes, dataset_name):
    """
    计算Dice系数
    """
    if y_pred.shape != y_true.shape:
        print("y_pred.shape: ", y_pred.shape, "y_true.shape: ", y_true.shape)
        raise ValueError("Shape mismatch between y_pred and y_true.")
    y_true = convert_labels(y_true, dataset_name)

    dice = 0.0
    for cls in range(num_classes):
        pred_cls = (y_pred == cls).astype(np.int32)
        true_cls = (y_true == cls).astype(np.int32)

        intersection = np.sum(pred_cls * true_cls)
        union = np.sum(pred_cls) + np.sum(true_cls)

        dice += (2. * intersection) / (union + 1e-6)
    return dice / num_classes


def iou_score(y_pred, y_true, num_classes, dataset_name):
    """
    计算IoU
    """
    if y_pred.shape != y_true.shape:
        print("y_pred.shape: ", y_pred.shape, "y_true.shape: ", y_true.shape)
        raise ValueError("Shape mismatch between y_pred and y_true.")
    y_true = convert_labels(y_true, dataset_name)

    iou = 0.0
    for cls in range(num_classes):
        pred_cls = (y_pred == cls).astype(np.int32)
        true_cls = (y_true == cls).astype(np.int32)

        intersection = np.sum(pred_cls * true_cls)
        union = np.sum(pred_cls) + np.sum(true_cls) - intersection

        iou += intersection / (union + 1e-6)
    return iou / num_classes


def pixel_error(y_pred, y_true, dataset_name):
    """
    计算像素误差
    """
    if y_pred.shape != y_true.shape:
        print("y_pred.shape: ", y_pred.shape, "y_true.shape: ", y_true.shape)
        raise ValueError("Shape mismatch between y_pred and y_true.")
    y_true = convert_labels(y_true, dataset_name)
    return 1.0 - np.mean(y_pred == y_true)


if __name__ == "__main__":
    # 模拟ACDC数据集标签
    y_true = np.random.choice([0, 85, 170, 255], (256, 256))
    y_pred = np.random.choice([0, 85, 170, 255], (256, 256))
    dataset_name = "ACDC"

    # 将预测转换为索引
    y_true = convert_labels(y_true, dataset_name)
    y_pred = convert_labels(y_pred, dataset_name)

    num_classes = 4  # ACDC有4类标签
    print("Dice Coefficient:", dice_coefficient(y_pred, y_true, num_classes, dataset_name))
    print("IoU:", iou_score(y_pred, y_true, num_classes, dataset_name))
    print("Pixel Error:", pixel_error(y_pred, y_true, dataset_name))

