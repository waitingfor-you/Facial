from PyQt5.QtCore import QThread, QDateTime
from PyQt5.QtCore import pyqtSignal


class TimerThread(QThread):
    signal=pyqtSignal(object)
    def __init__(self):
        super().__init__()
        self.running = True
    def run(self):
        while self.running:
            current_time = QDateTime.currentDateTime().toString("yyyy年MM月dd日\n hh:mm:ss ")
            self.signal.emit(current_time)
            QThread.sleep(1)
    def stop(self):
        self.running=False