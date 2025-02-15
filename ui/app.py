from ui.app_ui import ImageProcessApp
import sys
from PyQt5.QtCore import Qt, QThread, QTimer, QObject, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QFileDialog,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QScrollArea,
    QMessageBox,
    QSizePolicy,
)
from models.TableProcessModel_v2 import TableProcessModel
from Workers import ModelLoadWorker, ImageProcessWorker
from ui.logger import logger


class TableAnalysisTool(ImageProcessApp):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("TableAnalysisTool")
        self.model: TableProcessModel = None
        self.thread_ = None
        self.worker = None

    def load_model(self):
        super().load_model()
        self.thread_ = QThread()
        self.worker = ModelLoadWorker()
        self.worker.moveToThread(self.thread)

        # connect
        self.worker.model_loaded.connect(self.on_model_loaded)
        self.thread_.started.connect(self.worker.run)
        self.thread_.finished.connect(self.thread.deleteLater)

        self.thread_.start()

    # services releated
    def locate_table(self):
        super().locate_table()
        self.set_status_button_state("处理中")
        if not self.check_model_loaded() or not self.check_images_loaded():
            return
        self.worker = ImageProcessWorker(
            [self.image_list[self.current_idx]], self.model
        )
        self.thread_ = QThread()
        self.worker.moveToThread(self.thread_)
        self.set_status_button_state()
        self.set_status_button_state("处理")

        # connect
        self.worker.image_processed.connect(self.on_one_image_processed)

    def structure_table(self):
        super().structure_table()

    def recognize_table(self):
        super().recognize_table()

    def process_a4(self):
        return super().process_a4()

    def split_a3(self):
        super().split_a3()

    def process_a3(self):
        super().process_a3()

    def export_and_open_excel(self):
        super().export_and_open_excel()

    def export_and_open_history(self):
        super().export_and_open_history()

    def process_one_image(self):
        super().process_one_image()

    def process_images(self):
        super().process_images()

    # slot funcs
    @pyqtSlot(object)
    def on_model_loaded(self, model):
        QMessageBox.information(self, "info", "Model load sucessfully!")
        self.model = model
        self.thread.quit()

    def on_loaded(self):
        pass

    def on_located(self):
        pass

    def on_structured(self):
        pass

    def on_reced(self):
        pass

    def on_evaled(self):
        "locate + structure + rec"
        pass

    def on_one_image_processed(self):
        """显示当前进度"""
        logger.info(f"当前进度: {self.current_idx + 1}/{len(self.image_paths)}")
        pass

    def on_all_images_finished(self):
        QMessageBox.information(self, "info", "所有图片都已处理完成")
        # 启用按钮
        # 导出统计
        # 清理线程
        # 向用户确认是否需要清理队列
        # 清理模型统计得分历史
        pass

    # utils
    def check_model_loaded(self):
        if self.model is None:
            QMessageBox.information(
                self, "info", "Model has not been loaded successfully! Please wait"
            )
            return False
        return True

    def check_images_loaded(self):
        if len(self.image_paths) == 0:
            QMessageBox.information(self, "info", "No Image loaded! Please load images")
            return False
        return True


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TableAnalysisTool()
    window.show()
    sys.exit(app.exec_())
