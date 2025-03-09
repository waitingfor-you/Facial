import os.path
import shutil

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
    if not os.path.exists(image_path): return
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
def detect_and_draw(cache_path, model, device, labels):
    # 图像预处理
    image_files = [f for f in os.listdir(cache_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for image_path in image_files:
        original_image = cv2.imread(os.path.join(cache_path, image_path))
        processed_image, image_tensor = preprocess_image(os.path.join(cache_path, image_path))

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

    return

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

def clear_directory(folder_path):
    # 遍历文件夹中的所有文件和子文件夹
    # num = cnt
    for filename in os.listdir(folder_path):
        # if num == 0:
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # 删除文件或链接
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # 删除子文件夹
            # num -= 1
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
    file_count = len(os.listdir(folder_path))
    print(f"现在文件夹内容数量为:{file_count}")

# 主程序
if __name__ == "__main__":
    # 替换为实际图片路径和模型
    cache_path = '../cache'
    desired_fps = 10
    cap = cv2.VideoCapture(0)  # 参数 0 表示默认摄像头
    if not cap.isOpened():
        print("无法打开摄像头")
        exit()
    model_path = r'C:\Users\yu\Desktop\facial\utils\models\level\model-0.pt'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGGnet()  # 替换为实际模型类别数
    model.load_state_dict(torch.load(model_path, map_location=device))  # 加载权重
    model.to(device)  # 将模型移动到设备
    label = readLabel('../labels.txt')
    try:
        cnt = 0
        frames = []
        frame_paths = []
        frame_tmpl = os.path.join(cache_path, 'img_{:06d}.jpg')
        flag, frame = cap.read()
        # 检测并标注结果
        while flag:
            frames.append(frame)
            frame_path = frame_tmpl.format(cnt+1)
            frame_paths.append(frame_path)
            cv2.imwrite(frame_path, frame)
            # 继续读取下一帧
            cnt += 1
            flag, frame = cap.read()
            if cnt % 40 == 0:
                detect_and_draw(cache_path, model, device, label)
                if cnt == 120:
                    cnt = 0
                    clear_directory(cache_path)
                print("队列中的图片路径:")
                frame_paths = []
                frames = []
    finally:
        clear_directory(cache_path)

