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
from tools.Utils import *


class ProcessContext:
    def __init__(self) -> None:
        """对每一个图像的执行上下文进行保存"""
        self.action_chains = dict()
        self.action_state = dict()
        self.action_results = dict()
        pass


class TableAnalysisTool(ImageProcessApp):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("TableAnalysisTool")
        self.model: TableProcessModel = None
        self.thread_ = None
        self.worker = None
        self.load_model()
        self.action = None

    def setup_action(self, action):
        logger.info(f"Starting action: {action}")
        # logger.debug(
        #     f"Old thread status: s{hasattr(self, 'thread_') and self.thread_.isRunning()}"
        # )
        # if hasattr(self, 'thread_') and self.thread_.isRunning():
        #     self.thread_.quit()
        #     self.thread_.wait(1000)  # 最多等待1秒
        #     if self.thread_.isRunning():
        #         self.thread_.terminate()  # 强制终止
        #     self.thread_.deleteLater()
        #     del self.thread_
        # # 清理旧worker
        # if hasattr(self, 'worker'):
        #     self.worker.deleteLater()
        #     del self.worker

        self.set_status_button_state("处理中")
        self.action = action
        if not self.check_model_loaded():
            self.load_model()
            return
        if not self.check_images_loaded():
            return

        if self.is_batch_mode():
            self.worker = ImageProcessWorker(
                self.image_paths, self.model, action=action
            )
        else:
            self.worker = ImageProcessWorker(
                [self.image_paths[self.current_idx]],
                self.model,
                action=action,
            )
        self.thread_ = QThread() # QThread: Destroyed while thread is still running
        self.worker.moveToThread(self.thread_)

        # clean old connect
        # 在连接新信号前断开旧连接

        try:
            self.worker.show_next_signal.disconnect()
            self.worker.finished.disconnect()
        except TypeError:
            logger.error(f"worker fail to clean old connect", exc_info=True)

        # connect
        if not self.is_batch_mode():
            # self.worker.one_image_processed.connect(self.on_one_image_processed)
            self.worker.show_next_signal.connect(self.on_one_image_processed)
            # self.worker.finished.connect(self.on_one_image_processed)

            # self.worker.all_image_processed.connect(self.on_all_images_finished)
            self.worker.finished.connect(self.on_all_images_finished)

        else:
            self.worker.show_next_signal.connect(
                self.image_display_widget.show_next_image
            )
            self.worker.show_next_signal.connect(self.on_one_image_processed)

            # self.worker.one_image_processed.connect(self.on_one_image_processed)
            self.worker.all_image_processed.connect(self.on_all_images_finished)

        self.thread_.started.connect(self.worker.run)

    def load_model(self):
        super().load_model()
        self.thread_ = QThread()
        self.worker = ModelLoadWorker()
        self.worker.moveToThread(self.thread_)

        # connect
        self.worker.model_loaded.connect(self.on_model_loaded)
        self.thread_.started.connect(self.worker.run)
        self.thread_.finished.connect(self.thread_.deleteLater)

        self.thread_.start()

    # services releated
    def locate_table(self):
        super().locate_table()
        try:
            self.setup_action("locate")
            self.thread_.start()
            # 更新按钮状态
        except Exception:
            logger.info("locate fail")
            self.set_status_button_state("处理")

    def structure_table(self):
        super().structure_table()
        try:
            self.setup_action("structure")
            self.thread_.start()
            # 更新按钮状态
        except Exception:
            logger.info("structure fail")
            self.set_status_button_state("处理")

    def recognize_table(self):
        super().recognize_table()
        try:
            self.setup_action("ocr")
            self.thread_.start()
            # 更新按钮状态
        except Exception:
            logger.info("recognize fail")
            self.set_status_button_state("处理")

    def process_a4(self):
        super().process_a4()
        try:
            self.setup_action("a4_eval")
            self.thread_.start()
            # 更新按钮状态
        except Exception as e:
            logger.info(f"a4_eval fail, {e}")
            if self.check_images_loaded(info=False):
                self.set_status_button_state("处理")
            else:
                self.set_status_button_state("准备")

    def split_a3(self):
        super().split_a3()
        try:
            self.setup_action("a3_split")
            self.thread_.start()
            # 更新按钮状态
        except Exception as e:
            logger.info(f"a3_split fail, {e}")
            if self.check_images_loaded(info=False):
                self.set_status_button_state("处理")
            else:
                self.set_status_button_state("准备")

    def process_a3(self):
        try:
            super().process_a3()
            self.setup_action("a3_eval")
            self.thread_.start()
            # 更新按钮状态
        except Exception as e:
            logger.info(f"a3_eval fail, {e}")
            if self.check_images_loaded(info=False):
                self.set_status_button_state("处理")
            else:
                self.set_status_button_state("准备")

    def export_and_open_excel(self):
        super().export_and_open_excel()
        # 默认保存 提供路径打开文件夹即可
        open_folder()

    def export_and_open_history(self):
        super().export_and_open_history()
        # 默认保存 提供路径打开文件夹即可
        open_folder()
        # 如果没有保存 if
        self.model.a4table_score_eval_service.score_history_to_xlsx()

    # def process_one_image(self):
    #     super().process_one_image()

    # def process_images(self):
    #     super().process_images()

    # slot funcs
    @pyqtSlot(object)
    def on_model_loaded(self, model):
        QMessageBox.information(self, "info", "Model load sucessfully!")
        self.model = model
        self.thread_.quit()
        logger.debug("thread is quitting")
        self.thread_.wait(1000)


    def on_located(self):
        pass

    def on_structured(self):
        pass

    def on_reced(self):
        pass

    def on_evaled(self):
        "locate + structure + rec"
        pass

    @pyqtSlot(int)
    def on_one_image_processed(self, idx):
        """显示当前进度"""
        if self.batch_mode:
            logger.info(f"当前进度: {idx+1}/{len(self.image_paths)}")
            # 联动 image_list
            self.image_list.setCurrentRow(idx + 1)
        else:
            QMessageBox.information(self, "info", "选中的图片已处理完成")
            self.set_status_button_state("处理")

    def on_all_images_finished(self):
        if self.is_batch_mode():
            QMessageBox.information(self, "info", "所有图片都已处理完成")

        # 清理线程
        self.worker.deleteLater()
        self.thread_.quit()
        # self.thread_.wait()
        self.thread_.deleteLater()


        if self.action in ["a3_eval", "a4_eval"]:
            # 导出统计
            self.model.a4table_score_eval_service.score_history_to_xlsx()
            # 清理模型统计得分历史
            self.model.a4table_score_eval_service.score_history.clear()
        # 启用按钮
        self.set_status_button_state("处理")
        # 向用户确认是否需要清理队列

    # utils
    def check_model_loaded(self):
        if self.model is None:
            QMessageBox.information(
                self, "info", "Model has not been loaded successfully! Please wait"
            )
            return False
        return True


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TableAnalysisTool()
    window.show()
    sys.exit(app.exec_())
