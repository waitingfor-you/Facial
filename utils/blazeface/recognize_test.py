import cv2
import mediapipe as mp

# 初始化 MediaPipe Face Detection 模块
mp_face_detection = mp.solutions.face_detection

# 设置图片路径
image_path = r"C:\Users\yu\Desktop\facial\testPicture\img.png"  # 替换为你的图片路径

# 尝试加载图片
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
if image is None:
    print(f"无法读取图片，请检查路径是否正确：{image_path}")
    exit()

# 如果图片为 RGBA，转换为 RGB
if image.shape[-1] == 4:
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

# 转换为 RGB 图像（MediaPipe 使用 RGB）
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 加载模型并进行检测
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    # 检测面部
    results = face_detection.process(image_rgb)

    if results.detections:
        for idx, detection in enumerate(results.detections):
            # 获取边界框信息
            bounding_box = detection.location_data.relative_bounding_box
            width, height = image.shape[1], image.shape[0]  # 图像宽高
            x_min = int(bounding_box.xmin * width)
            y_min = int(bounding_box.ymin * height)
            box_width = int(bounding_box.width * width)
            box_height = int(bounding_box.height * height)

            print(f"人脸 {idx + 1}: x_min={x_min}, y_min={y_min}, width={box_width}, height={box_height}")
    else:
        print("未检测到人脸。")

    cv2.rectangle(image, (x_min, y_min), (x_min + box_width, y_min + box_height), (0, 255, 0), 2)

    # 显示结果
    cv2.imshow("Face Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
