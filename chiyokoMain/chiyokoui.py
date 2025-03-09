import datetime
import os
import sys
import queue

from collections import deque
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap, QPalette, QBrush, QImage
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QApplication, QLabel, QDesktopWidget, QTextEdit, \
    QLineEdit, QPushButton
from PyQt5 import QtGui

from chiyokoMain.CameraSet import CameraSet
from chiyokoMain.TimerThread import TimerThread
from emotinsDetect import *
from utils.recognize_picture import readLabel

filename = ''   # 用于调用摄像头，0为本地摄像头，但是无法直接输入192.168.1.100这样子的，需要加工
emotions = []   # 用于储存认为的表情
img_queue = queue.Queue()  # 用于储存摄像头传来的图像,这东西是不怕炸的，会自动
true_emotions = queue.Queue()  # 用于储存摄像头传来的图像,这东西是不怕炸的，会自动
emodict = { 'anger':'发怒',
        'disgust':'厌恶' ,
        'fear': '恐惧',
        'happy': '开心',
        'sad': '伤心',
        'surprised': '惊讶',
            'neutral' : '中性',
         'none':'无表情'
           }

path = r'..\assets\icons'

class Chiyoko(QWidget):
    def __init__(self):
        super().__init__()

        cache_path = '../cache'

        self.setWindowTitle("ChiyokoUI")
        self.setFixedSize(1618, 910)
        label = readLabel(r'..\labels.txt')
        self.center_window()  # 居中窗口的方法
        self.set_background(r"..\chiyokobg\zero.png")
        # 在这里面修改下就好了
        CameraSet(filename, emotions, img_queue, cache_path, true_emotions, label)

        self.create_widget()

        self.timerthread = TimerThread()
        self.timerthread.start()
        self.timerthread.signal.connect(self.update_time)


    def center_window(self):
        # 获取屏幕的可用几何信息
        screen_geometry = QDesktopWidget().availableGeometry()
        # 获取窗口自身的几何信息
        window_geometry = self.frameGeometry()
        # 将窗口移动到屏幕中央
        window_geometry.moveCenter(screen_geometry.center())
        self.move(window_geometry.topLeft())

    def create_widget(self):
        # 顶部区域
        self.top_frame = QLabel()
        self.top_frame.setFixedSize(1500, 500)
        self.top_frame.setStyleSheet('QLabel { border: 2px solid gray; }')
        self.top_frame.setAlignment(Qt.AlignVCenter | Qt.AlignCenter)
        pixmap = QPixmap(r'..\assets\selfemotion\normal.png')
        pixmap = pixmap.scaled(1500, 500, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.top_frame.setPixmap(pixmap)



        # 左下角 上面为摄像头，下面为时间
        self.camera_frame = QWidget()
        self.camera_label = QLabel()
        self.camera_label.setFixedSize(200, 170)

        self.camera_label.setStyleSheet('QLabel { border: 2px solid gray; }')

        self.timer_frame = QWidget()
        self.timer_label = QLabel()
        self.timer_label.setFixedSize(200, 190)
        font = QtGui.QFont()
        font.setBold(True)
        font.setFamily("华文楷体")
        font.setPointSize(14)
        self.timer_label.setFont(font)
        self.timer_label.setAlignment(Qt.AlignCenter)
        self.timer_label.setStyleSheet('QWidget { border: 2px solid gray; }')

        leftbottom_layout = QVBoxLayout()
        leftbottom_layout.addWidget(self.camera_label)
        leftbottom_layout.addWidget(self.timer_label)

        # 右下角 对话框
        self.dialog_frame = QWidget(self)
        self.dialog_frame.setFixedSize(1200, 360)
        self.dialog_frame.setStyleSheet('QWidget { border: 2px solid gray; }')

        # 创建对话框布局
        dialog_layout = QVBoxLayout()

        # 聊天记录展示
        self.chat_area = QTextEdit(self)
        self.chat_area.setReadOnly(True)
        dialog_layout.addWidget(self.chat_area)

        # 信息输入布局
        message_layout = QHBoxLayout()
        self.talker_input = QLineEdit(self)
        self.talker_input.setPlaceholderText('请输入...')
        self.talker_input_button = QPushButton('发送', self)
        self.talker_input_button.clicked.connect(self.send_message)
        message_layout.addWidget(self.talker_input)
        message_layout.addWidget(self.talker_input_button)

        # 将信息输入布局添加到对话框布局中
        dialog_layout.addLayout(message_layout)

        # 将对话框布局设置为对话框部件的布局
        self.dialog_frame.setLayout(dialog_layout)

        # 底部整体布局
        self.bottom_frame = QWidget()
        self.bottom_frame.setFixedSize(1450, 380)
        self.bottom_frame.setStyleSheet('QWidget { border: 2px solid gray; }')

        bottom_layout = QHBoxLayout()
        bottom_layout.addLayout(leftbottom_layout)
        bottom_layout.addWidget(self.dialog_frame)

        self.bottom_frame.setLayout(bottom_layout)

        # 主界面
        main_layout = QVBoxLayout()

        # 创建一个水平布局，将顶部区域居中显示
        center_top_layout = QHBoxLayout()
        center_top_layout.addStretch(1)  # 添加一个弹簧以将顶部区域推到中间
        center_top_layout.addWidget(self.top_frame)
        center_top_layout.addStretch(1)  # 再次添加一个弹簧以保持顶部区域居中

        main_layout.addLayout(center_top_layout)
        main_layout.addWidget(self.bottom_frame)

        self.setLayout(main_layout)

        self.timer2 = QTimer(self)
        self.timer2.timeout.connect(self.image_show)
        self.timer2.start(18)

        self.timer3 = QTimer(self)
        self.timer3.timeout.connect(self.image_update)
        self.timer3.start(18)

    def set_background(self, path):
        if path:
            pixmap = QPixmap(path)
            if pixmap.size() != (1618,910):
               pixmap = pixmap.scaled(1618,910, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            palette = QPalette()
            palette.setBrush(QPalette.Background, QBrush(pixmap))
            self.setPalette(palette)
        else:
            self.setStyleSheet('background: lightgray')

    def update_time(self, text):
        now = datetime.datetime.now()
        self.timer_label.setText(f'{text}')

    def image_show(self):
        if not img_queue.empty():
            frame = img_queue.get_nowait()  # 从队列中获取 QImgae 对象
            label_width = self.camera_label.width()  # 获取 label 的宽度
            label_height = self.camera_label.height()  # 获取 label 的高度

            # 调整图像尺寸
            scaled_img = frame.scaled(label_width, label_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            # 将 QImage 转换为 QPixmap 以便在 QLabel 中显示
            pixmap = QPixmap.fromImage(scaled_img)

            # 设置 QLabel 的 pixmap
            self.camera_label.setPixmap(pixmap)
    def image_update(self):
        if not true_emotions.empty():
            true_emotion = true_emotions.get_nowait()
            emopath = os.path.join(path, '{}.png'.format(true_emotion))
            pixmap = QPixmap(emopath)
            pixmap = pixmap.scaled(1500, 500, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            if pixmap.isNull():
                print(f"Failed to load image from {emopath}")
            self.top_frame.setPixmap(pixmap)
    def send_message(self):
        message = self.talker_input.text()
        if message:
            self.chat_area.append(f"User: {message}")
            self.talker_input.clear()


if __name__ == '__main__':
        try:
            app = QApplication(sys.argv)
            chiyoko = Chiyoko()
            chiyoko.show()
            sys.exit(app.exec_())
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            # 确保清理代码在程序退出时运行
            clear_directory(r'..\cache')
