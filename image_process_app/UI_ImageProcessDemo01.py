import sys
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
    QPushButton,
    QFileDialog,
    QSizePolicy,
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
from PyQt5.QtCore import Qt
from ImageDisplayWidget import ImageDisplayWidget
import os
from logger import logger


class ImageProcessingApp(QMainWindow):
    def __init__(self, config=None):
        super().__init__()

        # 设置主窗口属性
        self.setWindowTitle("Image Processing App")
        self.setMinimumSize(1920, 1080)
        self.setGeometry(100, 100, 800, 600)

        # 创建菜单栏
        self.create_menu_bar()

        # 创建工具栏
        self.create_tool_bar()

        # 创建图片列表
        self.files_dockwidget = QDockWidget("Files", self)
        self.image_list = QListWidget()
        self.files_dockwidget.setWidget(self.image_list)
        self.files_dockwidget.setMinimumWidth(200)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.files_dockwidget)

        self.image_list.itemClicked.connect(self.on_image_clicked)

        # 创建属性区
        self.properties_dockwidget = QDockWidget("Properties", self)
        self.properties_label = QLabel("Properties will be shown here.")
        self.properties_dockwidget.setWidget(self.properties_label)
        self.properties_dockwidget.setMinimumWidth(300)
        self.addDockWidget(Qt.RightDockWidgetArea, self.properties_dockwidget)

        # 创建状态栏
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")

        # 创建图像展示区域
        self.image_display_widget = ImageDisplayWidget()
        self.image_display_widget.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        self.setCentralWidget(self.image_display_widget)

        # 创建日志输出区
        self.log_widget = QDockWidget("Log", self)
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        self.log_widget.setWidget(self.log_text_edit)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.log_widget)

        self.log_text_edit.ensureCursorVisible()
        self.log_text_edit.verticalScrollBar().setValue(
            self.log_text_edit.verticalScrollBar().maximum()
        )

        # logic
        self.image_paths = []

        # 日志输出区
        logger.add_gui_handler(self.log_text_edit)
        logger.info("init done")
        logger.warning("a test warning")

    def create_menu_bar(self):
        menu_bar = self.menuBar()

        # 添加文件菜单
        file_menu = menu_bar.addMenu("File")
        open_file_action = QAction(QIcon("open.png"), "Open file", self)
        open_folder_action = QAction(QIcon("open_folder.png"), "Open folder", self)
        change_output_folder_action = QAction(
            QIcon("change_output_folder.png"), "Change output folder", self
        )
        exit_action = QAction("Exit", self)

        open_file_action.triggered.connect(self.open_file)
        open_folder_action.triggered.connect(self.open_folder)
        exit_action.triggered.connect(self.close)  # 绑定退出操作

        file_menu.addAction(open_file_action)
        file_menu.addAction(open_folder_action)
        file_menu.addAction(change_output_folder_action)
        file_menu.addAction(exit_action)

        views_menu = menu_bar.addMenu("Views")
        # 添加设置菜单
        settings_menu = menu_bar.addMenu("Settings")
        # 可以在这里添加设置相关的操作

        # 添加关于菜单
        about_menu = menu_bar.addMenu("About")
        license_action = QAction("Software License", self)
        about_menu.addAction(license_action)
        info_action = QAction("Software Information", self)
        about_menu.addAction(info_action)

        # 绑定关于菜单项的槽函数
        license_action.triggered.connect(self.show_license)
        info_action.triggered.connect(self.show_info)

    def show_license(self):
        # 这里可以打开一个新窗口或者对话框显示软件授权信息
        license_text = "This is the software license information."
        self.log_text_edit.setText(license_text)  # 使用日志区域显示信息

    def show_info(self):
        # 这里可以打开一个新窗口或者对话框显示软件信息
        info_text = (
            "Software Name: Image Processing App\nVersion: 1.0\nAuthor: Your Name"
        )
        self.log_text_edit.setText(info_text)  # 使用日志区域显示信息

    def create_tool_bar(self):
        tool_bar = QToolBar("Main ToolBar", self)
        self.addToolBar(tool_bar)

        # 添加表格定位按钮
        locate_table_action = QAction(QIcon("locate.png"), "Locate Table", self)
        tool_bar.addAction(locate_table_action)

        # 添加表格结构按钮
        structure_table_action = QAction(
            QIcon("structure.png"), "Structure Table", self
        )
        tool_bar.addAction(structure_table_action)

        # 添加表格识别按钮
        recognize_table_action = QAction(
            QIcon("recognize.png"), "Recognize Table", self
        )
        tool_bar.addAction(recognize_table_action)

        # 添加处理A3按钮
        process_a3_action = QAction(QIcon("a3.png"), "Process A3", self)
        tool_bar.addAction(process_a3_action)

        # 添加处理A4按钮
        process_a4_action = QAction(QIcon("a4.png"), "Process A4", self)
        tool_bar.addAction(process_a4_action)

        # 添加清除按钮
        clear_action = QAction(QIcon("clear.png"), "Clear", self)
        tool_bar.addAction(clear_action)

        # 绑定按钮的槽函数
        locate_table_action.triggered.connect(self.locate_table)
        structure_table_action.triggered.connect(self.structure_table)
        recognize_table_action.triggered.connect(self.recognize_table)
        process_a3_action.triggered.connect(self.process_a3)
        process_a4_action.triggered.connect(self.process_a4)
        clear_action.triggered.connect(self.clear)

    # 以下是对应的槽函数示例，您需要根据实际功能实现具体逻辑
    def open_file(self):
        # 打开文件对话框
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "", "Image Files (*.png *.jpg *.jpeg)"
        )
        if file_path:
            logger.info(f"open file {file_path}")
            self.image_list.clear()
            # 将选中的文件添加到图片列表
            self.image_list.addItem(file_path)
            self.image_paths = [file_path]
            self.image_display_widget.set_image_paths(self.image_paths)
            self.image_list.setCurrentRow(0)

    def open_folder(self):
        # 打开文件夹对话框
        folder_path = QFileDialog.getExistingDirectory(self, "Open Image Folder")
        if folder_path:
            logger.info(f"open folder {folder_path}")
            # 清空当前图片列表
            self.image_list.clear()
            # 遍历文件夹中的所有文件
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                # 如果是图片文件，则添加到图片列表
                if file_path.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.image_list.addItem(file_path)
                    self.image_paths.append(file_path)
                if self.image_paths:
                    self.image_display_widget.set_image_paths(self.image_paths)
                    self.image_list.setCurrentRow(0)

    def on_image_clicked(self, item):
        index = self.image_list.row(item)
        self.image_display_widget.set_current_index(index)
        self.image_list.setCurrentRow(index)

    def locate_table(self):
        self.status_bar.showMessage("Locating table...")
        # 实现表格定位逻辑

    def structure_table(self):
        self.status_bar.showMessage("Structuring table...")
        # 实现表格结构逻辑

    def recognize_table(self):
        self.status_bar.showMessage("Recognizing table...")
        # 实现表格识别逻辑

    def process_a3(self):
        self.status_bar.showMessage("Processing A3 page...")
        # 实现处理A3页面逻辑

    def process_a4(self):
        self.status_bar.showMessage("Processing A4 page...")
        # 实现处理A4页面逻辑

    def clear(self):
        self.status_bar.showMessage("Clearing...")
        # 实现清除逻辑，例如清空图片展示区域、日志等


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessingApp()
    window.show()
    sys.exit(app.exec_())
