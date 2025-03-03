# -*- coding: UTF-8 -*-
"""
@File    ：ui.py
@Author  ：Csy
@Date    ：2025/2/7 10:58 
@Bref    : 可以作为下游任务的统一UI框架, 子类继承即可 实现新的业务函数,工具栏 菜单栏已有基础实现可供参考
@Ref     :
TODO     :
         :
"""
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
    QFormLayout,
    QPushButton,
    QFileDialog,
    QSizePolicy,
    QMessageBox,
    QGroupBox,
    QComboBox,
    QDialog
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
from PyQt5.QtCore import Qt, QSize, QThread, QMutex

from ui.ImageDisplayWidget import ImageDisplayWidget
import os
import xlsxwriter, json  # 导入用于导出 Excel 的库，如果未安装请先安装：pip install xlsxwriter
from ui.logger import logger
from typing import Dict


class ImageProcessApp(QMainWindow):
    def __init__(self, config=None):
        super().__init__()

        # 设置主窗口属性
        self.setWindowTitle("Image Processing App")
        self.setMinimumSize(1920, 1080)
        self.setGeometry(100, 100, 800, 600)
        self.icons_dir = "ui/resources/icons/"

        # 创建菜单栏
        self.create_menu_bar()

        # 创建工具栏
        self.tool_bar = self.create_tool_bar()
        self.toolbar_actions = self.tool_bar.actions()
        self.locked_actions = self.init_lock_action()

        # 创建图片列表
        self.files_dockwidget = QDockWidget("Files", self)
        self.image_list = QListWidget()
        self.files_dockwidget.setWidget(self.image_list)
        self.files_dockwidget.setMinimumWidth(200)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.files_dockwidget)

        self.image_list.itemClicked.connect(self.on_image_clicked)

        # 创建属性区
        self.property_dockwidget = self.create_property_widget()

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
        self.log_widget = QDockWidget("Console", self)
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
        self.current_idx = 0
        self.batch_mode: bool = True
        self.mode_mutex = QMutex()  # for batch_mode access for mutli-thread
        self.actions_ = dict()  # store all actions

        # 日志输出区
        logger.add_gui_handler(self.log_text_edit)
        logger.info("init done")
        logger.warning("a test warning")

        self.model = None
        # thread
        self.thread_ = None
        self.worker = None

    def create_menu_bar(self):
        menu_bar = self.menuBar()

        # 添加文件菜单
        file_menu = menu_bar.addMenu("File")
        open_file_action = QAction(
            QIcon(self.icons_dir + "/open_file.png"), "Open file", self
        )
        open_folder_action = QAction(
            QIcon(self.icons_dir + "/open_dir.png"), "Open dir", self
        )
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

    def create_tool_bar(self):
        tool_bar = QToolBar("Main ToolBar", self)
        # tool_bar.resize(143,333)
        tool_bar.setMinimumHeight(36)
        tool_bar.setIconSize(QSize(40, 40))
        tool_bar.setStyleSheet("QToolBar{spacing:8px;}")

        self.addToolBar(tool_bar)
        open_file_action = QAction(
            QIcon(self.icons_dir + "/open_file.png"), "open one table image", self
        )
        open_file_action.setObjectName("openFileAction")
        tool_bar.addAction(open_file_action)

        open_dir_action = QAction(
            QIcon(self.icons_dir + "/open_dir.png"),
            "open a dir contains table images",
            self,
        )
        open_dir_action.setObjectName("openDirAction")
        tool_bar.addAction(open_dir_action)

        tool_bar.addSeparator()
        # 添加表格定位按钮
        locate_table_action = QAction(
            QIcon(self.icons_dir + "/table_locate.png"), "locate tables on images", self
        )
        locate_table_action.setObjectName("locateTableAction")
        tool_bar.addAction(locate_table_action)

        # 添加表格结构按钮
        structure_table_action = QAction(
            QIcon(self.icons_dir + "/table_structure.png"),
            "split tables' structure",
            self,
        )
        structure_table_action.setObjectName("structureTableAction")
        tool_bar.addAction(structure_table_action)

        # 添加表格识别按钮
        recognize_table_action = QAction(
            QIcon(self.icons_dir + "/table_rec.png"), "recognize table content", self
        )
        recognize_table_action.setObjectName("recognizeTableAction")
        tool_bar.addAction(recognize_table_action)
        tool_bar.addSeparator()

        # 添加处理A4按钮
        process_a4_action = QAction(
            QIcon(self.icons_dir + "/a4_eval3.png"), "处理A4单表问卷照片", self
        )
        process_a4_action.setObjectName("a4EvalAction")
        tool_bar.addAction(process_a4_action)
        tool_bar.addSeparator()

        # 添加处理A3按钮
        split_a3_action = QAction(
            QIcon(self.icons_dir + "/a3_split3.png"), "切分单张A3问卷", self
        )
        split_a3_action.setObjectName("a3SplitAction")
        tool_bar.addAction(split_a3_action)

        process_a3_action = QAction(
            QIcon(self.icons_dir + "/a3_eval3.png"), "处理A3正面多表问卷照片", self
        )
        process_a3_action.setObjectName("a3EvalAction")
        tool_bar.addAction(process_a3_action)

        process_a3_back_action = QAction(
            QIcon(self.icons_dir + "/a3_eval_back.png"), "处理A3反面多表问卷照片", self
        )
        process_a3_back_action.setObjectName("a3EvalBackAction")
        tool_bar.addAction(process_a3_back_action)
        tool_bar.addSeparator()

        # 添加清除按钮
        clear_action = QAction(
            QIcon(self.icons_dir + "/clear_queue.png"), "清空图像队列", self
        )
        clear_action.setObjectName("clearQueueAction")
        tool_bar.addAction(clear_action)

        tool_bar.addSeparator()
        # excel 相关
        # 一张图片上的excel
        one_img_excel_action = QAction(
            QIcon(self.icons_dir + "/excel_one_img.png"),
            "打开当前图像处理结果excel",
            self,
        )
        one_img_excel_action.setObjectName("oneImgExcelAtion")
        tool_bar.addAction(one_img_excel_action)

        history_excel_action = QAction(
            QIcon(self.icons_dir + "/excel_history.png"),
            "打开此次执行图像处理统计历史excel",
            self,
        )
        history_excel_action.setObjectName("historyExcelAtion")
        tool_bar.addAction(history_excel_action)

        # 绑定按钮的槽函数
        open_file_action.triggered.connect(self.open_file)
        open_dir_action.triggered.connect(self.open_folder)

        locate_table_action.triggered.connect(self.locate_table)
        structure_table_action.triggered.connect(self.structure_table)
        recognize_table_action.triggered.connect(self.recognize_table)

        process_a4_action.triggered.connect(self.process_a4)
        split_a3_action.triggered.connect(self.split_a3)
        process_a3_action.triggered.connect(self.process_a3)
        process_a3_back_action.triggered.connect(self.process_a3_back)

        clear_action.triggered.connect(self.clear_queue)

        one_img_excel_action.triggered.connect(self.export_and_open_excel)
        history_excel_action.triggered.connect(self.export_and_open_history)

        tool_bar.addSeparator()
        # single or batch
        mode_layout = QHBoxLayout()
        mode_widget = QWidget()

        mode_label = QLabel("处理\n模式")

        self.mode_combo = QComboBox()
        self.mode_combo.setBaseSize(40, 36)
        self.mode_combo.addItem("single")  # 添加 "单张模式" 选项
        self.mode_combo.addItem("batch")  # 添加 "批量模式" 选项
        self.mode_combo.setCurrentIndex(1)  # 索引 0 对应 "单张模式"
        # tool_bar.addWidget(self.mode_combo)
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_combo)
        mode_widget.setLayout(mode_layout)

        tool_bar.addWidget(mode_widget)
        self.mode_combo.currentIndexChanged.connect(self.mode_changed)

        return tool_bar

    def init_lock_action(self) -> Dict[str, QAction]:
        custom_action = [
            "openFileAction",
            "openDirAction",
            "locateTableAction",
            "structureTableAction",
            "recognizeTableAction",
            "a4EvalAction",
            "a3SplitAction",
            "a3EvalAction",
            "a3EvalBackAction",
            "clearQueueAction",
            "oneImgExcelAtion",
            "historyExcelAtion",
        ]
        short_des = [
            "open_file",
            "open_dir",
            "locate",
            "structure",
            "rec",
            "a4_eval",
            "a3_split",
            "a3_eval",
            "a3_eval_back",
            "clear_queue",
            "export_open_excel",
            "export_open_hisotry",
        ]
        locked_offset = 2
        locked_idx = 0
        des_action_dict = {}
        for action in self.toolbar_actions:
            if action.objectName() in custom_action[locked_offset:]:
                action.setDisabled(True)
                des_action_dict[short_des[locked_offset + locked_idx]] = action
                locked_idx += 1
        return des_action_dict

    def create_property_widget(self):
        properties_dockwidget = QDockWidget(
            "属性信息"
        )  # 修改 DockWidget 的标题为中文，更直观
        properties_dockwidget.setObjectName(
            "propertiesDockWidget"
        )  # 设置setObjectName，方便后续样式表或查找

        main_widget = QWidget()  # 创建一个主 Widget，用于容纳所有可折叠组件
        main_layout = QVBoxLayout(main_widget)  # 使用垂直布局

        # 1. 图像级别信息组件 (可折叠)
        image_groupbox = QGroupBox(
            "图像属性"
        )  # 使用 QGroupBox 创建可折叠组件，并设置标题
        image_groupbox.setObjectName(
            "imageGroupBox"
        )  # 设置setObjectName，方便后续样式表或查找
        image_layout = QFormLayout()  # 使用 FormLayout，更整齐地展示键值对信息

        self.image_width_label = QLabel("512 ")  # 使用 self.xxx 方便后续更新数值
        self.image_height_label = QLabel("512 ")
        self.image_type_label = QLabel("jpg ")

        image_layout.addRow(
            "宽度:", self.image_width_label
        )  # 使用 addRow 添加标签和对应的 QLabel
        image_layout.addRow("高度:", self.image_height_label)
        image_layout.addRow("类型:", self.image_type_label)
        image_groupbox.setLayout(image_layout)  # 设置 GroupBox 的布局
        main_layout.addWidget(image_groupbox)  # 将图像信息 GroupBox 添加到主布局中
        image_groupbox.setCheckable(True)  # 设置 GroupBox 可折叠
        image_groupbox.setChecked(True)  # 默认展开

        # 2. 表格定位信息组件 (可折叠)
        table_location_groupbox = QGroupBox("表格定位信息")
        table_location_groupbox.setObjectName(
            "tableLocationGroupBox"
        )  # 设置setObjectName，方便后续样式表或查找
        table_location_layout = QVBoxLayout()  # 使用垂直布局，因为 bbox 信息可能较长

        self.table_num_label = QLabel("定位到 n 张表格")
        self.table_bbox_label = QLabel(
            "定位表格 \n\n BBox1 [] \n BBox2 [] \n BBox3 []"
        )  # 使用 self.xxx 方便后续更新数值
        table_location_layout.addWidget(self.table_num_label)
        table_location_layout.addWidget(self.table_bbox_label)
        table_location_groupbox.setLayout(table_location_layout)
        main_layout.addWidget(table_location_groupbox)
        table_location_groupbox.setCheckable(True)
        table_location_groupbox.setChecked(True)  # 默认展开

        # 3. 表格划分信息组件 (可折叠)
        table_partition_groupbox = QGroupBox("表格划分信息")
        table_partition_groupbox.setObjectName(
            "tablePartitionGroupBox"
        )  # 设置setObjectName，方便后续样式表或查找
        table_partition_layout = QFormLayout()

        self.table_rows_label = QLabel("26 ")  # 使用 self.xxx 方便后续更新数值
        self.table_cols_label = QLabel("5 ")
        self.table_cells_label = QLabel("130 ")

        table_partition_layout.addRow("行数:", self.table_rows_label)
        table_partition_layout.addRow("列数:", self.table_cols_label)
        table_partition_layout.addRow("单元格数:", self.table_cells_label)
        table_partition_groupbox.setLayout(table_partition_layout)
        main_layout.addWidget(table_partition_groupbox)
        table_partition_groupbox.setCheckable(True)
        table_partition_groupbox.setChecked(True)  # 默认展开

        # 4. 表格 Cell 识别信息组件 (可折叠)
        cell_recognition_groupbox = QGroupBox("表格单元格识别信息")
        cell_recognition_groupbox.setObjectName(
            "cellRecognitionGroupBox"
        )  # 设置setObjectName，方便后续样式表或查找
        cell_recognition_layout = (
            QVBoxLayout()
        )  # 这里可以根据单元格信息的展示方式选择布局，例如表格、列表等

        self.cell_info_label = QLabel(
            "单元格信息将在此处展示。\n(例如：单元格内容列表或表格)"
        )  # 占位符，后续根据实际需求替换
        cell_recognition_layout.addWidget(self.cell_info_label)
        cell_recognition_groupbox.setLayout(cell_recognition_layout)
        main_layout.addWidget(cell_recognition_groupbox)
        cell_recognition_groupbox.setCheckable(True)
        table_partition_groupbox.setChecked(True)  # 默认展开

        # 导出
        export_button_layout = QHBoxLayout()  # 水平布局放置两个导出按钮

        export_excel_button = QPushButton("导出到 Excel")
        export_excel_button.setObjectName("exportExcelButton")
        export_excel_button.clicked.connect(
            self.export_properties_to_excel
        )  # 连接 Excel 导出函数
        export_button_layout.addWidget(export_excel_button)

        export_json_button = QPushButton("导出到 JSON")  # 新增 JSON 导出按钮
        export_json_button.setObjectName("exportJsonButton")
        export_json_button.clicked.connect(
            self.export_properties_to_json
        )  # 连接 JSON 导出函数
        export_button_layout.addWidget(export_json_button)

        main_layout.addLayout(export_button_layout)  # 添加按钮布局到主布局
        main_layout.setAlignment(export_button_layout, Qt.AlignRight)  # 按钮靠右

        properties_dockwidget.setWidget(main_widget)  # 设置 DockWidget 的主 Widget
        properties_dockwidget.setMinimumWidth(300)  # 设置最小宽度
        self.addDockWidget(
            Qt.RightDockWidgetArea, properties_dockwidget
        )  # 添加 DockWidget 到主窗口的右侧

        return properties_dockwidget  # 返回创建的 DockWidget 对象

    def open_file(self):
        # 打开文件对话框
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "", "Image Files (*.png *.jpg *.jpeg)"
        )
        if file_path:
            self.image_list.clear()
            self.image_paths.clear()
            logger.info(f"open file {file_path}")
            self.image_list.clear()
            # 将选中的文件添加到图片列表
            self.image_list.addItem(file_path)
            self.image_paths = [file_path]
            self.image_display_widget.set_image_paths(self.image_paths)
            self.image_list.setCurrentRow(self.current_idx)
            # enable action
            enable_action_des = [
                "a4_eval",
                "a3_split",
                "a3_eval",
                "a3_eval_back",
                "clear_queue",
                "export_open_excel",
            ]
            for des, action in self.locked_actions.items():
                if des in enable_action_des:
                    action.setEnabled(True)
            # update status button
            self.set_status_button_state("处理")

    def open_folder(self):
        # 打开文件夹对话框
        folder_path = QFileDialog.getExistingDirectory(self, "Open Image Folder")
        if folder_path:
            logger.info(f"open folder {folder_path}")
            # 清空当前图片列表
            self.image_list.clear()
            self.image_paths.clear()
            # 遍历文件夹中的所有文件
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                # 如果是图片文件，则添加到图片列表
                if file_path.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.image_list.addItem(file_path)
                    self.image_paths.append(file_path)
                if self.image_paths:
                    self.image_display_widget.set_image_paths(self.image_paths)
                    self.image_list.setCurrentRow(self.current_idx)
            # enable action
            enable_action_des = [
                "a4_eval",
                "a3_split",
                "a3_eval",
                "a3_eval_back",
                "clear_queue",
                "export_open_excel",
            ]
            for des, action in self.locked_actions.items():
                if des in enable_action_des:
                    action.setEnabled(True)
            # update status button
            self.set_status_button_state("处理")
        else:
            QMessageBox.information(self, "error", "文件夹打开失败")

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
        self.set_status_button_state("处理中")
        # # 实现处理A3页面逻辑
        # for i in range(100000):
        #     logger.info(i)
        # self.set_status_button_state("处理")

    def process_a3_back(self):
        self.status_bar.showMessage("Processing A3 back page...")
        self.set_status_button_state("处理中")

    def split_a3(self):
        self.status_bar.showMessage("Splitting A3 page...")
        # 实现处理A4页面逻辑

    def process_a4(self):
        self.status_bar.showMessage("Processing A4 page...")
        # 实现处理A4页面逻辑

    def export_and_open_excel(self):
        """
        检测文件是否存在
        默认每张图片处理流程包含导出文件 但是不打开
        """
        pass

    def export_and_open_history(self):
        """
        检测文件是否存在
        默认每次批量处理流程包含导出历史文件 但是不打开
        """
        pass

    def clear_queue(self):
        self.status_bar.showMessage("Clearing image queue...")
        # 清除图像队列
        if len(self.image_paths) == 0 or self.image_list.count() == 0:
            QMessageBox.information(self, "info", "当前图像队列为空")
        self.image_paths.clear()
        self.image_list.clear()
        # 清除图像显示窗口 当前显示图像 以及 set_image_paths 为 []
        self.image_display_widget.set_image_paths([])  # 内部已处理
        # lock action
        self.init_lock_action()
        # 图像处理按钮 显示区状态 设置为 preparing

        self.set_status_button_state("准备")
        QMessageBox.information(self, "info", "图像队列已清空，请重新选取图像")

    # --------------------- 其他 与图片处理相关的 后处理事件 ----------------
    def load_model(self):
        self.status_bar.showMessage("loading model ...")

    def on_image_clicked(self, item):
        self.current_idx = self.image_list.row(item)
        self.image_display_widget.set_current_index(self.current_idx)
        self.image_list.setCurrentRow(self.current_idx)
        logger.info(f"current image: {self.image_paths[self.current_idx]}")

    def process_one_image(self):
        """仅处理当前选中的图片"""
        pass

    def process_images(self):
        """处理队列中的所有图片"""
        pass

    def on_one_image_processed(self):
        """显示当前进度"""
        # logger.info(f"当前进度: {self.current_idx + 1}/{len(self.image_paths)}")
        pass

    def on_all_images_finished(self):
        # QMessageBox.information(self, "info", "所有图片都已处理完成")
        # # 启用按钮
        # # 导出统计
        # # 清理线程
        # # 向用户确认是否需要清理队列
        # # 清理模型统计得分历史
        pass

    # 更新 properties
    def on_located(self):
        pass

    def on_structured(self):
        pass

    def on_reced(self):
        pass

    def on_evaled(self):
        "locate + structure + rec"
        pass

    def mode_changed(self, index):
        selected_idx = self.mode_combo.currentIndex()
        selected_mode = self.mode_combo.currentText()
        # logger.info(f"current mode {selected_mode}")
        if selected_idx == 0:
            self.batch_mode = False
            logger.info("switch to single-shot processing mode")
            # self.image_display_widget.status_button.setEnabled(True)
            # self.image_display_widget.status_button2.setDisabled(True)
        elif selected_idx == 1:
            self.batch_mode = True
            logger.info("switch to batch processing mode")
            # self.image_display_widget.status_button.setDisabled(True)
            # self.image_display_widget.status_button2.setEnabled(True)
        if self.check_images_loaded():
            self.set_status_button_state("处理")

    # --------------------- 工具类 ----------------
    def export_properties_to_excel(self):
        """导出属性信息到 Excel 文件 (根据复选框选择性导出)"""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(
            self, "保存 Excel 文件", "", "Excel 文件 (*.xlsx)"
        )

        export_image_info_checkbox = self.property_dockwidget.widget().findChild(
            QGroupBox, "imageGroupBox"
        )
        export_table_location_checkbox = self.property_dockwidget.widget().findChild(
            QGroupBox, "tableLocationGroupBox"
        )
        export_table_partition_checkbox = self.property_dockwidget.widget().findChild(
            QGroupBox, "tablePartitionGroupBox"
        )
        export_cell_recognition_checkbox = self.property_dockwidget.widget().findChild(
            QGroupBox, "cellRecognitionGroupBox"
        )

        if file_path:
            try:
                workbook = xlsxwriter.Workbook(file_path)
                worksheet = workbook.add_worksheet("属性信息")

                data = []  # 用于存储要导出的数据

                # 根据复选框状态决定是否导出各部分信息
                if export_image_info_checkbox.isChecked():
                    data.extend(
                        [
                            ["图像级别信息", ""],
                            [
                                "宽度",
                                self.image_width_label.text().replace("宽度: ", ""),
                            ],
                            [
                                "高度",
                                self.image_height_label.text().replace("高度: ", ""),
                            ],
                            [
                                "类型",
                                self.image_type_label.text().replace("类型: ", ""),
                            ],
                            ["", ""],
                        ]
                    )
                if export_table_location_checkbox.isChecked():
                    data.extend(
                        [
                            ["表格定位信息", ""],
                            [
                                "表格 BBox",
                                self.table_bbox_label.text().replace("表格 BBox: ", ""),
                            ],
                            ["", ""],
                        ]
                    )
                if export_table_partition_checkbox.isChecked():
                    data.extend(
                        [
                            ["表格划分信息", ""],
                            [
                                "行数",
                                self.table_rows_label.text().replace("行数: ", ""),
                            ],
                            [
                                "列数",
                                self.table_cols_label.text().replace("列数: ", ""),
                            ],
                            [
                                "单元格数",
                                self.table_cells_label.text().replace("单元格数: ", ""),
                            ],
                            ["", ""],
                        ]
                    )
                if export_cell_recognition_checkbox.isChecked():
                    data.extend(
                        [
                            ["表格单元格识别信息", ""],
                            [
                                "单元格信息",
                                self.cell_info_label.text().replace(
                                    "单元格信息将在此处展示。\n(例如：单元格内容列表或表格)",
                                    "",
                                ),
                            ],
                        ]
                    )

                row = 0
                col = 0
                for section in data:
                    worksheet.write(row, col, section[0])
                    worksheet.write(row, col + 1, section[1])
                    row += 1

                workbook.close()
                QMessageBox.information(
                    self, "导出成功", f"属性信息已导出到 Excel: {file_path}"
                )

            except Exception as e:
                QMessageBox.critical(self, "导出失败", f"导出 Excel 文件失败: {str(e)}")

    def export_properties_to_json(self):
        """导出属性信息到 JSON 文件 (根据复选框选择性导出)"""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(
            self, "保存 JSON 文件", "", "JSON 文件 (*.json)"
        )
        export_image_info_checkbox = self.property_dockwidget.widget().findChild(
            QGroupBox, "imageGroupBox"
        )
        export_table_location_checkbox = self.property_dockwidget.widget().findChild(
            QGroupBox, "tableLocationGroupBox"
        )
        export_table_partition_checkbox = self.property_dockwidget.widget().findChild(
            QGroupBox, "tablePartitionGroupBox"
        )
        export_cell_recognition_checkbox = self.property_dockwidget.widget().findChild(
            QGroupBox, "cellRecognitionGroupBox"
        )

        if file_path:
            try:
                properties_data = {}  # 使用字典存储 JSON 数据

                if export_image_info_checkbox.isChecked():
                    properties_data["图像级别信息"] = {  # 使用 section name 作为 key
                        "宽度": self.image_width_label.text().replace(
                            "宽度: ", ""
                        ),  # 去除标签前缀
                        "高度": self.image_height_label.text().replace("高度: ", ""),
                        "类型": self.image_type_label.text().replace("类型: ", ""),
                    }
                if export_table_location_checkbox.isChecked():
                    properties_data["表格定位信息"] = {
                        "表格 BBox": self.table_bbox_label.text().replace(
                            "表格 BBox: ", ""
                        )
                    }
                if export_table_partition_checkbox.isChecked():
                    properties_data["表格划分信息"] = {
                        "行数": self.table_rows_label.text().replace("行数: ", ""),
                        "列数": self.table_cols_label.text().replace("列数: ", ""),
                        "单元格数": self.table_cells_label.text().replace(
                            "单元格数: ", ""
                        ),
                    }
                if export_cell_recognition_checkbox.isChecked():
                    properties_data["表格单元格识别信息"] = {
                        "单元格信息": self.cell_info_label.text().replace(
                            "单元格信息将在此处展示。\n(例如：单元格内容列表或表格)", ""
                        )
                    }

                with open(
                    file_path, "w", encoding="utf-8"
                ) as f:  # 使用 utf-8 编码，支持中文等字符
                    json.dump(
                        properties_data, f, indent=4, ensure_ascii=False
                    )  # indent=4 格式化输出, ensure_ascii=False 支持中文

                QMessageBox.information(
                    self, "导出成功", f"属性信息已导出到 JSON: {file_path}"
                )

            except Exception as e:
                QMessageBox.critical(self, "导出失败", f"导出 JSON 文件失败: {str(e)}")

    def set_status_button_state(self, status):
        # update status button
        state_text = "准备中..."
        state_text2 = "准备中..."
        disabled = True
        color = None
        if status == "准备":
            state_text = "准备中..."
            state_text2 = "准备中..."
            disabled = True
        elif status == "处理":
            state_text = "处理"
            state_text2 = "批量处理"
            disabled = False
        elif status == "处理中":
            state_text = "处理中..."
            state_text2 = "批量处理中..."
            color = "green"
            disabled = True
        else:
            raise Exception("不支持的状态")

        self.image_display_widget.update_button_state(
            self.image_display_widget.status_button,
            state_text if not self.is_batch_mode() else state_text2,
            disabled=disabled,
            color=color,
        )

        # if status == "准备":
        #     self.image_display_widget.update_button_state(
        #         self.image_display_widget.status_button,
        #         state_text,
        #         disabled=disabled,
        #         color=color,
        #     )
        #     # self.image_display_widget.update_button_state(
        #     #     self.image_display_widget.status_button2,
        #     #     state_text2,
        #     #     disabled=disabled,
        #     #     color=color,
        #     # )
        # else:
        #     if not self.batch_mode:
        #         self.image_display_widget.update_button_state(
        #             self.image_display_widget.status_button,
        #             state_text,
        #             disabled=disabled,
        #             color=color,
        #         )
        #     else:
        #         self.image_display_widget.update_button_state(
        #             self.image_display_widget.status_button2,
        #             state_text2,
        #             disabled=disabled,
        #             color=color,
        #         )

    def check_images_loaded(self, info=True):
        if len(self.image_paths) == 0:
            if info:
                QMessageBox.information(
                    self, "info", "No Image loaded! Please load images"
                )
            return False
        return True

    def is_batch_mode(self):
        self.mode_mutex.lock()
        mode = self.batch_mode
        self.mode_mutex.unlock()
        return mode

    def show_license(self):
        # 这里可以打开一个新窗口或者对话框显示软件授权信息
        license_text = "This is the software license information."
        QMessageBox.information(self, "license", license_text)
        self.log_text_edit.setText(license_text)  # 使用日志区域显示信息

    def show_info(self):
        # 这里可以打开一个新窗口或者对话框显示软件信息
        info_text = (
            "Software Name: Image Processing App\nVersion: 1.0\nAuthor: Your Name"
        )
        QMessageBox.information(self, "info", info_text)
        self.log_text_edit.setText(info_text)  # 使用日志区域显示信息


# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = ImageProcessingApp()
#     window.show()
#     sys.exit(app.exec_())
