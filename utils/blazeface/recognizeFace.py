import cv2
import mediapipe as mp

def recognizeFace(image_path):
    mp_face_detection = mp.solutions.face_detection
    image = cv2.imread(image_path)
    if image is None:
        return

    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
    bbox = [x_min, y_min, box_width, box_height]
    return bbox


