import argparse
import shutil
import threading
import time
from collections import Counter

import cv2
import numpy as np
import os
import argparse

import torch
from PyQt5.QtGui import QImage, QPixmap
from torchvision import transforms

from model.VGGnet.joint import VGGnet
from utils.blazeface.recognizeFace import recognizeFace

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"




parser = argparse.ArgumentParser()
parser.add_argument("--source", type=int, default=0, help="data source, 0 for camera 1 for video")
parser.add_argument("--video_path", type=str, default=None)
opt = parser.parse_args()

if opt.source == 1 and opt.video_path is not None:
    filename = opt.video_path
else:
    filename = None

def generate_faces(face_img, img_size=48):
    """
    将探测到的人脸进行增广
    :param face_img: 灰度化的单个人脸图
    :param img_size: 目标图片大小
    :return:
    """

    face_img = face_img / 255.
    face_img = cv2.resize(face_img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    resized_images = list()
    resized_images.append(face_img)
    resized_images.append(face_img[2:45, :])
    resized_images.append(face_img[1:47, :])
    resized_images.append(cv2.flip(face_img[:, :], 1))

    for i in range(len(resized_images)):
        resized_images[i] = cv2.resize(resized_images[i], (img_size, img_size))
        resized_images[i] = np.expand_dims(resized_images[i], axis=-1)
    resized_images = np.array(resized_images)
    return resized_images

def load_model(device, model_path):
    """
    加载本地模型
    :return:
    """
    model = VGGnet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model

def Camera_progress(emotions, image_queue, cache_path, true_emotions, label, filename=0):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device, r'..\utils\models\level\model-50.pt')
    capture = cv2.VideoCapture(filename)
    if filename == '':
        capture = cv2.VideoCapture(0)

    cnt = 0
    frames = []
    frame_paths = []
    frame_tmpl = os.path.join(cache_path, 'img_{:06d}.jpg')
    flag, frame = capture.read()

    while flag:
        frames.append(frame)
        frame_path = frame_tmpl.format(cnt + 1)
        frame_paths.append(frame_path)
        cv2.imwrite(frame_path, frame)
        frame = cv2.cvtColor(cv2.resize(frame, (800, 600)), cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        image_queue.put(convert_to_Qt_format)
        # 继续读取下一帧
        cnt += 1
        flag, frame = capture.read()
        # if cnt == 40:
        #
        #     cnt = 0
        if cnt % 40 == 0:
            True_emotinon = detect_and_draw(cache_path, model, device, label)
            true_emotions.put(True_emotinon)
            if cnt == 120:
                cnt = 0
                clear_directory(cache_path)
            print("队列中的图片路径:")
            frame_paths = []
            frames = []
        # frameForShow = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # # 用于在ui左下角展示
        # frame = cv2.cvtColor(cv2.resize(frame, (800, 600)), cv2.COLOR_BGR2RGB)
        # h, w, ch = frame.shape
        # bytes_per_line = ch * w
        # convert_to_Qt_format = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        # image_queue.put(convert_to_Qt_format)
        # emotion = detect_and_draw(cache_path, model, device, emotions)
        # true_emotions.put(emotion)

        # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # faces = blaze_detect(frame)
        #
        # if faces is not None and len(faces) > 0:
        #     for (x, y, w, h) in faces:
        #         face = frame_gray[y: y + h, x: x + w]  # 脸部图片
        #         faces = generate_faces(face)
        #         results = model.predict(faces)
        #         result_sum = np.sum(results, axis=0).reshape(-1)
        #         label_index = np.argmax(result_sum, axis=0)
        #         emotions.append(index2emotion(label_index))
        #         if len(emotions) >= 100:
        #             counter = Counter(emotions)
        #             true_emotions.put(counter.most_common(1)[0][0])
        #             emotions.clear()


        key = cv2.waitKey(30)  # 等待30ms，返回ASCII码

    # 如果输入esc则退出循环
        if key == 27:
            break


    capture.release()  # 释放摄像头
    cv2.destroyAllWindows()  # 销毁窗口

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

def detect_and_draw(cache_path, model, device, labels):
    # 图像预处理
    items = os.listdir(cache_path)
    if len(items) == 40:
        true_labels = []
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
            true_labels.append(class_label)
            # 绘制检测结果

        true_label = Counter(true_labels).most_common(1)[0][0]
        clear_directory(cache_path)

    return true_label

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