import os
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QListWidget,
    QLabel,
    QDockWidget,
    QTextEdit,
    QAction,
    QToolBar,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QPushButton,
    QFileDialog,
    QSizePolicy,
    QMessageBox,
    QGroupBox,
    QComboBox,
    QDialog,
    QLineEdit,
    QDialogButtonBox,
    QButtonGroup,
    QRadioButton,
)
from PyQt5.QtGui import (
    QIcon,
    QPixmap,
    QImage,
    QKeyEvent,
    QPainter,
    QPen,
    QPalette,
    QImageReader,
)
from PyQt5.QtCore import Qt, QSize, QThread, QMutex, pyqtSignal
import numpy as np


class PathSelectDialog(QDialog):
    path_selected = pyqtSignal(dict)  # 定义信号用于传递路径数据

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("A3背面表格关联信息")
        self.setWindowIcon(QIcon("ui/resources/icons/folder.png"))
        self.setFixedSize(500, 270)

        self.init_ui()
        self.assoicate_back = True

    def init_ui(self):
        layout = QVBoxLayout()

        # 人员信息表路径
        self.xlsx_path = QLineEdit()
        self.xlsx_path.setPlaceholderText("请选择人员信息表路径")
        btn_xlsx = QPushButton("选择Excel文件")
        btn_xlsx.setIcon(QIcon("ui/resources/icons/excel.png"))
        btn_xlsx.clicked.connect(self.select_xlsx)

        # 工作文件夹路径
        self.folder_path = QLineEdit()
        self.folder_path.setPlaceholderText("请选择工作文件夹路径")
        btn_folder = QPushButton("选择工作目录")
        btn_folder.setIcon(QIcon("ui/resources/icons/folder.png"))
        btn_folder.clicked.connect(self.select_folder)

        # 按钮容器
        btn_container = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self
        )
        btn_container.accepted.connect(self.accept)
        btn_container.rejected.connect(self.reject)

        # 布局组织
        form_layout = QFormLayout()
        form_layout.addRow(
            "人员信息表:", self.create_path_row(self.xlsx_path, btn_xlsx)
        )
        form_layout.addRow(
            "工作文件夹:", self.create_path_row(self.folder_path, btn_folder)
        )
        self.side_group = QButtonGroup(self)
        self.front_radio = QRadioButton("正面", self)
        self.back_radio = QRadioButton("反面", self)

        self.back_radio.setChecked(True)
        self.side_group.addButton(self.front_radio)
        self.side_group.addButton(self.back_radio)
        # 创建水平布局容器
        radio_layout = QHBoxLayout()
        radio_layout.addWidget(self.front_radio)
        radio_layout.addWidget(self.back_radio)
        radio_layout.addStretch(1)  # 添加弹性空间

        form_layout.addRow("关联面选择：", radio_layout)

        layout.addLayout(form_layout)
        layout.addWidget(btn_container)
        self.setLayout(layout)

    def create_path_row(self, line_edit, button):
        container = QWidget()
        hbox = QHBoxLayout()
        hbox.addWidget(line_edit)
        hbox.addWidget(button)
        container.setLayout(hbox)
        return container

    def select_xlsx(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择人员信息表", "", "person_info.xlsx"
        )
        if path.endswith("person_info.xlsx"):
            self.xlsx_path.setText(path)

    def select_folder(self):
        self.get_selected_side()
        path_ = QFileDialog.getExistingDirectory(self, "选择工作目录")
        if path_:
            files = [
                file for file in os.listdir(path_) if os.path.isfile(path_ + "/" + file)
            ]
            front_collect_table_names = [
                "A3_LEFT_NO_1_TABLE",
                "A3_RIGHT_NO_1_TABLE",
                "A3_RIGHT_NO_2_TABLE",
                "A3_RIGHT_NO_3_TABLE",
            ]
            back_collect_table_name = ["A3_BACK_NO_2_TABLE", "A3_BACK_NO_3_TABLE"]
            back_state = []
            front_state = []
            for file in files:
                if file.startswith(back_collect_table_name[0]) or file.startswith(
                    back_collect_table_name[1]
                ):
                    back_state.append(True)
                if (
                    file.startswith(front_collect_table_names[0])
                    or file.startswith(front_collect_table_names[1])
                    or file.startswith(front_collect_table_names[2])
                    or file.startswith(front_collect_table_names[3])
                ):
                    front_state.append(True)

            if len(back_state) >= len(back_collect_table_name) and np.all(
                np.array(back_state)
            ):
                if self.assoicate_back:
                    self.folder_path.setText(path_)
                else:
                    QMessageBox.warning(
                        self,
                        "error",
                        "选择的关联面 与 工作文件夹属性不一致，请重新选择",
                    )
            if len(front_state) >= len(front_collect_table_names) and np.all(
                np.array(front_state)
            ):
                if self.assoicate_back:
                    QMessageBox.warning(
                        self,
                        "error",
                        "选择的关联面 与 工作文件夹属性不一致，请重新选择",
                    )
                else:
                    self.folder_path.setText(path_)

    def get_selected_side(self) -> str:
        """获取当前选中的检测面"""
        if self.front_radio.isChecked():
            self.assoicate_back = False
        elif self.back_radio.isChecked():
            self.assoicate_back = True

    def get_paths(self):
        return {
            "xlsx": self.xlsx_path.text(),
            "folder": self.folder_path.text(),
            "associate_back": self.assoicate_back,
        }

    def accept(self):
        """重写确认按钮事件"""
        if not self.validate_paths():
            QMessageBox.warning(self, "路径错误", "请正确选择两个路径！")
            return
        self.path_selected.emit(self.get_paths())
        super().accept()

    def validate_paths(self):
        return all(
            [
                os.path.isfile(self.xlsx_path.text()),
                os.path.isdir(self.folder_path.text()),
            ]
        )
