# -*- coding: UTF-8 -*-
"""
@File    ：ui.py
@Author  ：Csy
@Date    ：2024/12/12 10:58 
@Bref    :
@Ref     :
TODO     :
         :
"""
import sys
import time

from PyQt5.QtCore import Qt, QThread, QTimer, QObject, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog, QHBoxLayout, QVBoxLayout, QLabel, \
    QScrollArea, QMessageBox, QSizePolicy
from PyQt5.QtGui import QPixmap, QImage, QIcon
from Settings import *
from TableProcess import TableProcessModel
from Utils import *
from Workers import *


class ImageProcessingApp(QWidget):
    def __init__(self):
        super().__init__()
        # window setting
        self.setWindowTitle("问卷表格智能统分助手")   # 设置窗口名
        self.setWindowIcon(QIcon("./resources/icon16x16.png"))

        # self.setBaseSize(640, 480)
        # self.setMinimumSize(1080, 720)
        # self.setMinimumSize(640, 480)
        self.setMinimumSize(1280, 960)

        # 创建按钮
        self.open_file_button = QPushButton("打开图像文件", self)
        self.open_folder_button = QPushButton("打开图像文件夹", self)
        self.process_button = QPushButton("开始处理", self)

        self.show_progress_label = QLabel("当前进度: 0/0")
        self.show_progress_label.setAlignment(Qt.AlignCenter)

        self.show_label = QLabel("图像状态")
        self.show_label.setAlignment(Qt.AlignCenter)
        self.show_label.setWordWrap(True)

        self.image_label = QLabel("图像显示")
        # self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        # self.image_label.setScaledContents(True)
        self.image_label.setAlignment(Qt.AlignCenter)
        # default image
        self.image_label.setPixmap(self.scale_image(
            # QPixmap.fromImage(QImage('./table.jpg'))))
            QPixmap.fromImage(QImage('./resources/home_screen.png'))))

        self.image_scroll_area = QScrollArea()
        # self.image_scroll_area.resize(400, 400)
        self.image_scroll_area.setWidget(self.image_label)
        self.image_scroll_area.setAlignment(Qt.AlignCenter)

        # 连接信号与槽
        self.open_file_button.clicked.connect(self.open_file)
        self.open_folder_button.clicked.connect(self.open_folder)
        self.process_button.clicked.connect(self.process_images_v2)

        # 设置布局
        hbox = QHBoxLayout()
        hbox.addStretch()
        hbox.addWidget(self.open_file_button)
        hbox.addWidget(self.open_folder_button)
        hbox.addWidget(self.process_button)
        hbox.addStretch()

        vbox = QVBoxLayout()
        vbox.addWidget(self.show_progress_label)
        vbox.addWidget(self.show_label)
        # vbox.addWidget(self.image_label, 3)
        vbox.addWidget(self.image_scroll_area, 3)
        vbox.addStretch()
        vbox.addLayout(hbox, 1)
        vbox.addStretch()

        self.setLayout(vbox)

        # service
        self.images_need_process = []
        self.images_processed = []
        self.images_temp_during_process = {}  # 中间图像 绘制row 绘制col 绘制cell image_name:[]
        self.next_idx = 0
        # self.contoller = TableProcessController()
        # self.contoller.processing_done.connect(self.update_ui)
        # self.contoller.processing_start.connect(self.update_ui)

        self.model: TableProcessModel = None
        self.thread = None
        self.worker = None

        # load model by thread
        self.load_model()

    def open_file(self):
        # 打开图像文件
        file_name, _ = QFileDialog.getOpenFileName(
            self, "打开图像", "", "Image Files (*.png *.jpg *.bmp)")
        if file_name:
            # 加载图像并显示
            pixmap = QPixmap(file_name)
            # ... (将pixmap显示到QLabel或其他控件)
            self.images_need_process.extend([file_name])
            self.show_image(pixmap)
            self.show_progress_label.setText(
                f"当前进度: 0/{len(self.images_need_process)}")

    def open_folder(self):
        # 打开图像文件夹
        folder_path = QFileDialog.getExistingDirectory(self, "打开图像文件夹")
        if folder_path:
            # print(type(folder_path),folder_path)
            image_files = [folder_path + "/" + image for image in os.listdir(folder_path) if
                           os.path.splitext(image)[-1] in IMAGE_EXTS]
            self.images_need_process.extend(image_files)
            QMessageBox.information(
                self, "info", f'当前文件下一共{len(image_files)}张图片待处理')
            self.show_progress_label.setText(
                f"当前进度: 0/{len(self.images_need_process)}")
        else:
            QMessageBox.information(self, "error", "文件夹打开失败")

    def show_image(self, image_pixmap: QPixmap, info=True):
        # self.image_label.setPixmap(image_pixmap)
        # self.image_label.adjustSize()
        self.image_label.setPixmap(self.scale_image(image_pixmap))
        if info:
            QMessageBox.information(self, "info", "加载图像成功")

    def scale_image(self, image_pixmap: QPixmap):
        label_width = self.image_label.width()
        label_height = self.image_label.height()
        # label_width = 400
        # label_height = 300
        # print(label_width, label_height)
        image_width = image_pixmap.width()
        image_height = image_pixmap.height()

        # 计算缩放比例
        scale_factor = min(label_width / image_width,
                           label_height / image_height)
        scaled_width = int(image_width * scale_factor)
        scaled_height = int(image_height * scale_factor)
        # scaled_width = 300
        # scaled_height = 400
        # print(scaled_width, scaled_height)
        # 缩放图像
        pixmap = image_pixmap.scaled(
            scaled_width, scaled_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        return pixmap

    def load_model(self):
        self.thread = QThread()
        self.worker = ModelLoadWorker()
        self.worker.moveToThread(self.thread)

        # connect
        self.worker.model_loaded.connect(self.on_model_loaded)
        self.thread.started.connect(self.worker.run)
        self.thread.finished.connect(self.thread.deleteLater)
        #
        self.thread.start()

    @pyqtSlot(object)
    def on_model_loaded(self, model):
        QMessageBox.information(self, "info", "Model load sucessfully!")
        self.model = model
        self.thread.quit()

    def process_images_v2(self):
        if self.model is None:
            QMessageBox.information(
                self, 'info', "Model has not been loaded successfully! Please wait")
            return
        if len(self.images_need_process) == 0:
            QMessageBox.information(
                self, 'info', "No Image loaded! Please load images")
            return
        self.process_button.setEnabled(False)
        self.thread = QThread()
        self.worker = ImageProcessWorker(
            self.images_need_process, self.model, log=True)
        self.worker.moveToThread(self.thread)

        self.worker.image_processed.connect(self.update_ui)
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.show_signal.connect(self.load_image_on_screen)

        self.thread.started.connect(self.worker.run)

        self.thread.start()

    def process_images(self):
        global table_process
        if table_process is None:
            table_process = TableProcessModel()
        for image_path in self.images_need_process:
            self.show_image(QPixmap(image_path), info=False)
            print('processing ', image_path, '--->', end=' ')
            table_process.run(image_path)
            print('done')
        print('all images processed')

    @pyqtSlot(str)
    def update_ui(self, state_info):
        self.show_label.setText(state_info)
        time.sleep(0.5)

    @pyqtSlot(int)
    def load_image_on_screen(self, idx):
        if idx < len(self.images_need_process):
            self.show_progress_label.setText(
                f"当前进度: {idx + 1}/{len(self.images_need_process)}")
            self.show_image(QPixmap(self.images_need_process[idx]), info=False)
            time.sleep(1)

    @pyqtSlot()
    def on_processing_finished(self):
        """处理完成后恢复 UI"""
        # self.show_label.setText("All images processed!")
        # add only for opening single image
        self.show_progress_label.setText(
            f"当前进度: {len(self.images_need_process)}/{len(self.images_need_process)}")
        QMessageBox.information(self, 'info', "所有图片都已处理完成")
        self.process_button.setEnabled(True)
        self.model.score_eval.score_history_to_xlsx()

        # 清理线程
        self.worker.deleteLater()
        self.thread.quit()
        self.thread.wait()
        self.thread.deleteLater()
        # self.thread = None

        # 清理图像队列
        open_folder(os.path.split(self.images_need_process[0])[0])
        self.images_need_process.clear()

        # 清理统计得分历史
        self.model.score_eval.score_history.clear()

    def show_instructions(self):
        QMessageBox.information(
            self,
            "Info",
            '''
        性能：cuda~3.5s/张; cpu~4.5s/张
        使用说明：
        1. 需要等待提示模型加载完成后方可使用
        2. 所有需要处理图片放在同一个文件夹下
        3. 如果启用中间图像缓存, 放置在图像所在文件夹 cache 中
        4. 输出xlsx文件放置在图片同级目录下
        5. 每处理一次,输出一次处理过程中所有表格的得分统计
        ''',
        )


# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     window = ImageProcessingApp()
#     window.show()
#     QTimer.singleShot(100,window.show_instructions)
#     sys.exit(app.exec_())
