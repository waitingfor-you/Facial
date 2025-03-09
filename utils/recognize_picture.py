import os.path

import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from model.VGGnet.joint import VGGnet
from utils.blazeface.recognizeFace import recognizeFace


# 图像预处理函数
def preprocess_image(image_path):
    # 读取图片
    image = cv2.imread(image_path)
    bbox = recognizeFace(image_path)
    height = bbox[3]
    width = bbox[2]
    # 截取中间位置的正方形区域
    x_min = bbox[0]
    y_min = bbox[1]
    cropped_image = image[y_min:y_min+height, x_min:x_min+width]

    # 转为单通道灰度图像
    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    # 调整大小为 48x48
    resized_image = cv2.resize(gray_image, (48, 48))

    # 转换为 PyTorch 张量
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
    image_tensor = transform(resized_image).unsqueeze(0)  # 增加 batch 维度
    return resized_image, image_tensor

# 检测并绘制结果
def detect_and_draw(image_path, model, device, labels):
    # 图像预处理
    original_image = cv2.imread(image_path)
    processed_image, image_tensor = preprocess_image(image_path)

    # 使用模型预测
    image_tensor = image_tensor.to(device)
    class_index = model.forward(image_tensor)
    _, index = torch.max(class_index, dim=1)
    index = index.item()
    class_label = labels[index]

    # 绘制检测结果
    annotated_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)  # 转为彩色以便绘制文字
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(original_image, class_label, (10, 20), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    return original_image

def readLabel(labelpath):
    if not os.path.exists(labelpath):
        print("文件路径不存在！！！")
        return
    with open(labelpath, 'r') as file:
        lines = file.readlines()

    cleaned_lines = []
    for line in lines:
        line = line.replace(' ', '')
        line = ''.join([char for char in line if not char.isdigit()])
        cleaned_lines.append(line.strip())
    return cleaned_lines
# 主程序
if __name__ == "__main__":
    # 替换为实际图片路径和模型
    image_path = r"C:\Users\yu\Desktop\facial\testPicture\img.png"
    model_path = r'C:\Users\yu\Desktop\facial\utils\models\level\model-0.pt'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGGnet()  # 替换为实际模型类别数
    model.load_state_dict(torch.load(model_path, map_location=device))  # 加载权重
    model.to(device)  # 将模型移动到设备
    label = readLabel('../labels.txt')
    # 检测并标注结果
    result_image = detect_and_draw(image_path, model, device, label)

    # 显示和保存结果
    cv2.imshow("Detection Result", result_image)
    cv2.imwrite("annotated_image.jpg", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
