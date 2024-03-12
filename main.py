import sys
import cv2
import numpy as np
import process
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.uic import loadUi
from PyQt5.QtGui import QPixmap, QImage

class Window(QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        loadUi('GUImain.ui', self)
        
        with open("style.css", "r") as css:
            self.setStyleSheet(css.read())
        
        self.face_decector, self.eye_detector, self.detector = process.init_cv()
        
        self.startButton.clicked.connect(self.start_webcam)
        self.stopButton.clicked.connect(self.stop_webcam)
        
        self.camera_is_running = False
        self.previous_right_keypoints = None
        self.previous_left_keypoints = None
        self.previous_right_blob_area = None
        self.previous_left_blob_area = None

    def start_webcam(self):
        if not self.camera_is_running:
            # 尝试使用DirectShow。如果失败，则回退到默认摄像头。
            self.capture = cv2.VideoCapture(cv2.CAP_DSHOW)
            if not self.capture.isOpened():
                self.capture = cv2.VideoCapture(0)  # sometimes drops error
                if not self.capture.isOpened():
                    print("Can't open web camera")
                    return
            
            self.camera_is_running = True
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(2)  # 设置定时器每2毫秒触发一次

    def stop_webcam(self):
        if self.camera_is_running:
            self.capture.release()
            self.timer.stop()
            self.camera_is_running = False

    def update_frame(self):
        # 更新帧处理逻辑
        ret, base_image = self.capture.read()
        if ret:
            self.display_image(base_image)

            processed_image = cv2.cvtColor(base_image, cv2.COLOR_RGB2GRAY)

            # 使用process模块的方法处理图像
            face_frame, face_frame_gray, left_eye_estimated_position, right_eye_estimated_position, _, _ = process.detect_face(
                base_image, processed_image, self.face_decector)

            if face_frame is not None:
                # 检测并处理眼睛
                left_eye_frame, right_eye_frame, left_eye_frame_gray, right_eye_frame_gray = process.detect_eyes(face_frame,
                                                                                                                 face_frame_gray,
                                                                                                                 left_eye_estimated_position,
                                                                                                                 right_eye_estimated_position,
                                                                                                                 self.eye_detector)

                # 处理右眼图像
                if right_eye_frame is not None and self.rightEyeCheckbox.isChecked():
                    right_eye_threshold = self.rightEyeThreshold.value()
                    right_keypoints, self.previous_right_keypoints, self.previous_right_blob_area = self.get_keypoints(
                        right_eye_frame, right_eye_frame_gray, right_eye_threshold,
                        previous_area=self.previous_right_blob_area,
                        previous_keypoint=self.previous_right_keypoints)
                    process.draw_blobs(right_eye_frame, right_keypoints)

                    right_eye_frame = np.require(right_eye_frame, np.uint8, 'C')
                    self.display_image(right_eye_frame, window='right')

                # 处理左眼图像
                if left_eye_frame is not None and self.leftEyeCheckbox.isChecked():
                    left_eye_threshold = self.leftEyeThreshold.value()
                    left_keypoints, self.previous_left_keypoints, self.previous_left_blob_area = self.get_keypoints(
                        left_eye_frame, left_eye_frame_gray, left_eye_threshold,
                        previous_area=self.previous_left_blob_area,
                        previous_keypoint=self.previous_left_keypoints)
                    process.draw_blobs(left_eye_frame, left_keypoints)

                    left_eye_frame = np.require(left_eye_frame, np.uint8, 'C')
                    self.display_image(left_eye_frame, window='left')

            # draws keypoints on pupils on main window
            if self.pupilsCheckbox.isChecked():
                self.display_image(base_image)

    def get_keypoints(self, frame, frame_gray, threshold, previous_keypoint, previous_area):
        # 获取关键点
        keypoints = process.process_eye(frame_gray, threshold, self.detector,
                                        prevArea=previous_area)
        if keypoints:
            previous_keypoint = keypoints
            previous_area = keypoints[0].size
        else:
            keypoints = previous_keypoint
        return keypoints, previous_keypoint, previous_area

    def display_image(self, img, window='main'):
        # 显示图像
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:
            if img.shape[2] == 4:  # RGBA
                qformat = QImage.Format_RGBA8888
            else:  # RGB
                qformat = QImage.Format_RGB888

        out_image = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        out_image = out_image.rgbSwapped()
        if window == 'main':  # main window
            self.baseImage.setPixmap(QPixmap.fromImage(out_image))
            self.baseImage.setScaledContents(True)
        elif window == 'left':  # left eye window
            self.leftEyeBox.setPixmap(QPixmap.fromImage(out_image))
            self.leftEyeBox.setScaledContents(True)
        elif window == 'right':  # right eye window
            self.rightEyeBox.setPixmap(QPixmap.fromImage(out_image))
            self.rightEyeBox.setScaledContents(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    window.setWindowTitle("Gaze Estimation test")
    window.show()
    sys.exit(app.exec_())
