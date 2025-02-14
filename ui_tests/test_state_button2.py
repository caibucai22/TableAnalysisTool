import sys
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
)
from PyQt5.QtCore import QTimer, Qt


class ButtonStateWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("按钮状态示例")
        self.setGeometry(300, 300, 400, 200)

        self.process_button = QPushButton("开始处理")
        self.batch_process_button = QPushButton("开始批量处理")

        self.process_status_label = QLabel("状态: 待处理")
        self.batch_process_status_label = QLabel("状态: 待处理")

        self.process_button.clicked.connect(self.start_process)
        self.batch_process_button.clicked.connect(self.start_batch_process)

        main_layout = QVBoxLayout()

        process_layout = QHBoxLayout()
        process_layout.addWidget(self.process_button)
        process_layout.addWidget(self.process_status_label)

        batch_process_layout = QHBoxLayout()
        batch_process_layout.addWidget(self.batch_process_button)
        batch_process_layout.addWidget(self.batch_process_status_label)

        main_layout.addLayout(process_layout)
        main_layout.addLayout(batch_process_layout)

        self.setLayout(main_layout)

        self.process_timer = QTimer(self)  # 模拟处理过程的定时器
        self.process_timer.timeout.connect(self.finish_process)
        self.batch_process_timer = QTimer(self)  # 模拟批量处理过程的定时器
        self.batch_process_timer.timeout.connect(self.finish_batch_process)

    def start_process(self):
        self.update_button_state(
            self.process_button, self.process_status_label, "准备中...", disabled=True
        )
        QTimer.singleShot(
            1000,
            lambda: self.update_button_state(
                self.process_button,
                self.process_status_label,
                "处理中...",
                text_processing="处理中...",
                color="blue",
            ),
        )  # 1秒后进入 "处理中"
        self.process_timer.start(2000)  # 2秒后模拟处理完成

    def finish_process(self):
        self.process_timer.stop()
        self.update_button_state(
            self.process_button,
            self.process_status_label,
            "已处理",
            text_normal="重新处理",
            disabled=False,
            color="green",
        )

    def start_batch_process(self):
        self.update_button_state(
            self.batch_process_button,
            self.batch_process_status_label,
            "准备中...",
            disabled=True,
        )
        QTimer.singleShot(
            1000,
            lambda: self.update_button_state(
                self.batch_process_button,
                self.batch_process_status_label,
                "批量处理中...",
                text_processing="批量处理中...",
                color="blue",
            ),
        )  # 1秒后进入 "批量处理中"
        self.batch_process_timer.start(3000)  # 3秒后模拟批量处理完成

    def finish_batch_process(self):
        self.batch_process_timer.stop()
        self.update_button_state(
            self.batch_process_button,
            self.batch_process_status_label,
            "已批量处理",
            text_normal="重新批量处理",
            disabled=False,
            color="green",
        )

    def update_button_state(
        self,
        button,
        status_label,
        status_text,
        text_normal=None,
        text_processing=None,
        disabled=False,
        color=None,
    ):
        """更新按钮和状态标签的显示状态"""
        if text_normal:
            button.setText(text_normal)  # 恢复正常文本
        else:
            button.setText(status_text)  # 设置为状态文本

        status_label.setText(f"状态: {status_text}")
        button.setEnabled(not disabled)  # 设置按钮的启用/禁用状态

        if color:
            button.setStyleSheet(
                f"background-color: {color}; color: white;"
            )  # 设置背景颜色和文字颜色
        else:
            button.setStyleSheet("")  # 移除自定义样式，恢复默认样式


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ButtonStateWindow()
    window.show()
    sys.exit(app.exec_())
