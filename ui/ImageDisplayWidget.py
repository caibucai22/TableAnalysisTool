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
    QLabel,
    QSizePolicy,
    QScrollArea,
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
    QTransform,
    QWheelEvent,
    QMouseEvent,
)
from PyQt5.QtCore import Qt, QSize, QPoint, QRectF
from logger import logger

# ImageScaler.py
from PyQt5.QtCore import QPoint, QSize, Qt
from PyQt5.QtGui import QPixmap, QTransform


class ImageScaler:
    def __init__(self):
        self._original_pixmap = QPixmap()
        self._current_scale = 1.0
        self._min_scale = 0.1
        self._max_scale = 10.0
        self._offset = QPoint(0, 0)
        self._container_size = QSize()

    def initialize(self, pixmap: QPixmap, container_size: QSize):
        """初始化缩放器"""
        self._original_pixmap = pixmap
        self._container_size = container_size
        self._current_scale = self._calculate_fit_scale()
        self._offset = QPoint(0, 0)

    def _calculate_fit_scale(self) -> float:
        """计算自适应缩放比例"""
        if self._original_pixmap.isNull():
            return 1.0

        width_ratio = self._container_size.width() / self._original_pixmap.width()
        height_ratio = self._container_size.height() / self._original_pixmap.height()
        return min(width_ratio, height_ratio)

    def zoom(self, factor: float, focus_point: QPoint = QPoint()):
        """执行缩放操作
        :param factor: 缩放系数（例如1.1表示放大10%）
        :param focus_point: 以该点为中心的相对坐标（QPoint）
        """
        # 计算新缩放比例
        new_scale = self._current_scale * factor
        new_scale = max(self._min_scale, min(new_scale, self._max_scale))

        if new_scale == self._current_scale:
            return

        # 计算焦点相对位置
        focus_rel_x = (focus_point.x() - self._offset.x()) / self._current_scale
        focus_rel_y = (focus_point.y() - self._offset.y()) / self._current_scale

        # 更新缩放比例
        self._current_scale = new_scale

        # 调整偏移量保持焦点位置
        new_offset_x = focus_point.x() - focus_rel_x * self._current_scale
        new_offset_y = focus_point.y() - focus_rel_y * self._current_scale
        self._offset = QPoint(int(new_offset_x), int(new_offset_y))

    def reset(self):
        """重置到自适应缩放状态"""
        self._current_scale = self._calculate_fit_scale()
        self._offset = QPoint(0, 0)

    def pan(self, delta: QPoint):
        """平移视图
        :param delta: 鼠标移动的像素差值
        """
        # # 计算有效平移范围
        # img_width = self.image_label.width()
        # img_height = self.image_label.height()
        # viewport = self.scroll_area.viewport()

        # # 限制最大偏移量
        # max_x = max(0, img_width - viewport.width())
        # max_y = max(0, img_height - viewport.height())

        # new_x = min(max(0, self.image_scaler._offset.x() + delta.x()), max_x)
        # new_y = min(max(0, self.image_scaler._offset.y() + delta.y()), max_y)

        # self._offset = QPoint(new_x, new_y)

        # v1
        self._offset += delta

    def get_transformed_pixmap(self) -> QPixmap:
        """获取变换后的图像"""
        if self._original_pixmap.isNull():
            return QPixmap()

        # v2 TODO: 还未处理好
        # # 创建足够大的画布
        # canvas_size = (
        #     self._original_pixmap.size() * self.current_scale
        # )
        # canvas_size += QSize(100, 100)  # 添加边距

        # final_pixmap = QPixmap(canvas_size)
        # final_pixmap.fill(Qt.transparent)

        # painter = QPainter(final_pixmap)
        # painter.drawPixmap(
        #     self.view_offset,
        #     self._original_pixmap.scaled(
        #         self._original_pixmap.size()
        #         * self.current_scale,
        #         Qt.KeepAspectRatio,
        #         Qt.SmoothTransformation,
        #     ),
        # )
        # painter.end()

        # v1
        # 应用缩放变换
        transform = QTransform().scale(self._current_scale, self._current_scale)
        scaled_pixmap = self._original_pixmap.transformed(
            transform, Qt.SmoothTransformation
        )

        # 应用偏移变换
        final_pixmap = QPixmap(scaled_pixmap.size())
        final_pixmap.fill(Qt.transparent)
        painter = QPainter(final_pixmap)
        painter.drawPixmap(self._offset, scaled_pixmap)
        painter.end()

        return final_pixmap

    @property
    def current_scale(self) -> float:
        return self._current_scale

    @property
    def view_offset(self) -> QPoint:
        return self._offset


class ImageDisplayWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # --------------------------ui + connect------------------------
        # 创建布局
        main_layout = QVBoxLayout(self)

        # 创建图像显示区域布局
        self.image_layout = QHBoxLayout()
        main_layout.addLayout(self.image_layout, 4)
        main_layout.addSpacing(10)
        # 创建切换按钮布局
        control_layout = QHBoxLayout()
        main_layout.addLayout(control_layout, 1)

        # 图像显示 QScrollArea + qlabel
        self.scroll_area = QScrollArea()
        # self.scroll_area.setWidgetResizable(True) # v1
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setAlignment(Qt.AlignCenter)
        self.scroll_area.setViewportMargins(0, 0, 0, 0)
        self.scroll_area.setStyleSheet(
            """
            QScrollArea { border: none; }
            QScrollBar:vertical { width: 12px; }
            QScrollBar:horizontal { height: 12px; }
        """
        )

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setScaledContents(False)  # 必须关闭自动缩放
        # self.scroll_area.setWidget(self.image_label)

        # 容器widget
        self.scroll_content = QWidget()
        self.scroll_layout = QHBoxLayout(self.scroll_content)
        self.scroll_layout.addWidget(self.image_label)
        self.scroll_area.setWidget(self.scroll_content)

        background_image_path = "background.png"  # 背景图片路径
        if self.load_background_image(background_image_path):
            self.image_label.setPixmap(QPixmap(background_image_path))
        else:
            # 如果背景图片加载失败，设置纯色背景
            self.set_background_color(Qt.lightGray)

        self.image_layout.addWidget(self.scroll_area)

        control_layout.addStretch(10)
        # 创建切换按钮
        self.prev_button = QPushButton("<", self)
        self.prev_button.setFixedSize(60, 40)
        control_layout.addWidget(self.prev_button)
        # 状态按钮
        self.status_button = QPushButton(
            # "preparing", self
            "准备中",
            self,
        )  # preparing prepared processing processed
        self.status_button.setFixedSize(100, 40)
        self.status_button.setEnabled(False)
        control_layout.addWidget(self.status_button)

        self.status_button2 = QPushButton(
            # "preparing", self
            "准备中",
            self,
        )  # preparing prepared processing processed
        self.status_button2.setFixedSize(100, 40)
        self.status_button2.setEnabled(False)
        control_layout.addWidget(self.status_button2)

        self.next_button = QPushButton(">", self)
        self.next_button.setFixedSize(60, 40)
        control_layout.addWidget(self.next_button)
        control_layout.addStretch(10)

        # self.reset_btn = QPushButton("reset view", self)
        # self.reset_btn.clicked.connect(self.reset_view)
        # control_layout.addWidget(self.reset_btn)

        # 连接按钮信号
        self.prev_button.clicked.connect(self.show_previous_image)
        self.next_button.clicked.connect(self.show_next_image)

        # -------------------------------- var related to logic ------------------------------
        self.current_index = 0
        self.image_paths = []  # 图片路径
        self.rectangles = []  # 用于存储矩形框信息的列表
        self.last_mouse_pos = None
        self.image_scaler = ImageScaler()

        # -------------------------------- event ------------------------------
        self.setMouseTracking(True)
        self.image_label.setMouseTracking(True)
        self.scroll_area.setMouseTracking(True)
        # 安装事件过滤器以监听鼠标滚轮和键盘事件
        self.installEventFilter(self)

    # -------------------------------- init & set funcs ------------------------------
    def set_image_paths(self, paths):
        self.image_paths = paths
        self.current_index = 0
        self.load_image()

    def set_current_index(self, index):
        self.current_index = index
        self.load_image()

    def set_background_color(self, color):
        """设置纯色背景"""
        palette = self.image_label.palette()
        palette.setColor(QPalette.Window, color)
        self.image_label.setAutoFillBackground(True)
        self.image_label.setPalette(palette)

    def load_background_image(self, path):
        """尝试加载背景图片，成功返回True，失败返回False"""
        if not path:
            return False
        pixmap = QPixmap(path)
        if pixmap.isNull():
            return False
        return True

    def load_image(self):
        if self.current_index < len(self.image_paths):
            # image = QImage(self.image_paths[self.current_index])
            # 使用QImageReader处理方向
            try:
                reader = QImageReader(self.image_paths[self.current_index])
                reader.setAutoTransform(True)  # 启用自动转换
                image = reader.read()
                pixmap = QPixmap.fromImage(image)
                # 获取实际显示区域尺寸
                container_size = self.scroll_area.size()
                if container_size.width() == 0 or container_size.height() == 0:
                    container_size = self.size()  # 使用widget尺寸作为备用

                self.image_scaler.initialize(pixmap=pixmap, container_size=container_size)
                logger.info(f"scale_factor is {self.image_scaler.current_scale}")
                self.update_display()
                logger.info("scale to fit display")
            except:
                logger.error("load image failed")
        else:
            self.image_label.clear()
            # self.status_label.setText("No more images")

    def show_previous_image(self):
        self.current_index = max(0, self.current_index - 1)
        self.load_image()
        self.parent().image_list.setCurrentRow(self.current_index)  # 添加联动

    def show_next_image(self):
        self.current_index = min(len(self.image_paths) - 1, self.current_index + 1)
        self.load_image()
        self.parent().image_list.setCurrentRow(self.current_index)  # 添加联动

    def add_rectangle(self, rect, pen_color=Qt.red, pen_width=2):
        """添加一个矩形框到绘制列表"""
        self.rectangles.append(
            {"rect": rect, "color": pen_color, "width": pen_width / self.scale_factor}
        )
        self.update()  # 触发重绘

    # ------------------------------------- event --------------------------------
    def eventFilter(self, obj, event):
        if obj == self:
            if event.type() == event.Wheel:
                if event.modifiers() & Qt.ControlModifier:
                    if event.angleDelta().y() > 0:
                        self.image_scaler._current_scale *= 1.1
                    else:
                        self.image_scaler._current_scale = max(
                            0.1, self.image_scaler._current_scale * 0.9
                        )
                    self.update_display()
                return True
            elif event.type() == event.KeyPress:
                if event.key() == Qt.Key_Left:
                    self.show_previous_image()
                    return True
                elif event.key() == Qt.Key_Right:
                    self.show_next_image()
                    return True
        return super().eventFilter(obj, event)

    def resizeEvent(self, event):
        """窗口大小变化时重新计算缩放"""
        super().resizeEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        if event.modifiers() & Qt.ControlModifier:
            # v1
            # zoom_factor = 1.2 if event.angleDelta().y() > 0 else 0.8

            # # 获取鼠标相对图像的位置
            # mouse_pos = event.pos() - self.image_label.pos()
            # # 执行缩放
            # self.image_scaler.zoom(zoom_factor, mouse_pos)
            # self.update_display()

            # v2
            # 获取正确的鼠标位置
            mouse_pos = self.scroll_area.viewport().mapFromGlobal(event.globalPos())
            self.image_scaler.zoom(
                zoom_factor=1.2 if event.angleDelta().y() > 0 else 0.8,
                focus_point=mouse_pos - self.image_label.pos(),
            )
            self.update_display()
        else:
            super().wheelEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.last_mouse_pos = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if event.buttons() & Qt.LeftButton:
            delta = event.pos() - self.last_mouse_pos
            self.image_scaler.pan(delta)
            self.update_display()
            self.last_mouse_pos = event.pos()
        super().mouseMoveEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.image_label.pixmap():
            return

        painter = QPainter(self.image_label.pixmap())
        painter.setRenderHint(QPainter.TextAntialiasing)

        # 应用变换矩阵
        transform = QTransform()
        transform.translate(
            self.image_scaler._offset.x(), self.image_scaler._offset.y()
        )
        transform.scale(
            self.image_scaler.current_scale, self.image_scaler.current_scale
        )
        painter.setTransform(transform)

        # 绘制所有存储的矩形框
        for rect_info in self.rectangles:
            pen = QPen(rect_info["color"], rect_info["width"])
            painter.setPen(pen)
            painter.drawRect(rect_info["rect"])

        painter.end()

    def update_process_button_state(
        self, button: QPushButton, status, text_normal=None, disabled=False, color=None
    ):
        if text_normal:
            button.setText(text_normal)  # 恢复正常文本
        else:
            button.setText(status)  # 设置为状态文本
        button.setEnabled(not disabled)

        if color:
            button.setStyleSheet(
                f"background-color: {color}; color: white;"
            )  # 设置背景颜色和文字颜色
        else:
            button.setStyleSheet("")  # 移除自定义样式，恢复默认样式

    # ------------------------------------- utils --------------------------------
    def update_display(self):
        """更新图像显示"""
        transformed_pixmap = self.image_scaler.get_transformed_pixmap()

        # 计算实际显示尺寸
        img_width = transformed_pixmap.width()
        img_height = transformed_pixmap.height()

        # 设置标签尺寸
        self.image_label.setFixedSize(img_width, img_height)
        self.image_label.setPixmap(transformed_pixmap)

        # self.image_label.setPixmap(transformed_pixmap)
        # self.image_label.resize(transformed_pixmap.size())

        # 更新容器尺寸
        self.scroll_content.setMinimumSize(img_width, img_height)

        # 调整滚动条位置
        viewport = self.scroll_area.viewport()
        visible_center = viewport.rect().center()
        content_center = self.image_label.rect().center()
        self.scroll_area.ensureVisible(
            content_center.x(),
            content_center.y(),
            visible_center.x(),
            visible_center.y(),
        )

    def reset_view(self):
        self.image_scaler.reset()
        self.update_display()

    def update_button_state(
        self,
        button,
        status_text,
        text_normal=None,
        disabled=False,
        color=None,
    ):
        """更新按钮和状态标签的显示状态"""
        if text_normal:
            button.setText(text_normal)  # 恢复正常文本
        else:
            button.setText(status_text)  # 设置为状态文本

        # status_label.setText(f"状态: {status_text}")
        button.setEnabled(not disabled)  # 设置按钮的启用/禁用状态

        if color:
            button.setStyleSheet(
                f"background-color: {color}; color: white;"
            )  # 设置背景颜色和文字颜色
        else:
            button.setStyleSheet("")  # 移除自定义样式，恢复默认样式
