import sys
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QVBoxLayout,
    QLabel,
    QHBoxLayout,
)
from PyQt5.QtCore import QTimer


class StatusWindow(QWidget):
    def __init__(self):
        super().__init__()

        # 初始化窗口
        self.setWindowTitle("状态处理器")
        self.setGeometry(100, 100, 400, 200)

        # 状态标签
        self.status_label = QLabel("状态: 未开始", self)

        # 按钮
        self.process_button = QPushButton("处理", self)
        self.batch_button = QPushButton("批量处理", self)

        # 禁用批量处理按钮
        self.batch_button.setEnabled(False)

        # 布局管理
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.status_label)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.process_button)
        button_layout.addWidget(self.batch_button)
        self.layout.addLayout(button_layout)

        self.setLayout(self.layout)

        # 连接按钮事件
        self.process_button.clicked.connect(self.start_processing)
        self.batch_button.clicked.connect(self.start_batch_processing)

        # 计时器（模拟处理过程的时间延迟）
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_status)

        self.processing_steps = 0  # 用于跟踪当前的处理步骤
        self.batch_steps = 0  # 批量处理的步骤数

    def update_status(self):
        # 更新状态的逻辑
        if self.processing_steps == 0:
            self.status_label.setText("状态: preparing")
        elif self.processing_steps == 1:
            self.status_label.setText("状态: processing")
        elif self.processing_steps == 2:
            self.status_label.setText("状态: processed")
            self.timer.stop()

    def start_processing(self):
        # 开始处理过程
        self.processing_steps = 0
        self.timer.start(1000)  # 每秒更新一次状态

        self.process_button.setEnabled(False)  # 处理过程中禁用按钮
        self.batch_button.setEnabled(False)  # 禁用批量处理按钮

    def start_batch_processing(self):
        # 批量处理过程
        self.batch_steps = 5  # 假设我们批量处理5次
        self.processing_steps = 0
        self.timer.start(1000)

        self.process_button.setEnabled(False)
        self.batch_button.setEnabled(False)

        self.timer.timeout.connect(self.batch_processing_step)

    def batch_processing_step(self):
        if self.batch_steps > 0:
            self.processing_steps = 0
            self.timer.start(1000)
            self.batch_steps -= 1
        else:
            self.status_label.setText("批量处理完成")
            self.timer.stop()
            self.process_button.setEnabled(True)
            self.batch_button.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StatusWindow()
    window.show()
    sys.exit(app.exec_())
