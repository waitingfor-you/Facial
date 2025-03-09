import threading


from emotinsDetect import Camera_progress

class CameraSet:
    def __init__(self, filename, emotions, image_queue, cache_path, true_emotions, label):
        Camera_Thread = threading.Thread(target=Camera_progress, args=(emotions, image_queue, cache_path, true_emotions, label, filename))
        Camera_Thread.start()