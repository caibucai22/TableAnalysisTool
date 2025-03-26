import os.path

from ui.app_ui import ImageProcessApp
import sys, glob
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
    QAction,
)
from PyQt5.QtGui import QIcon
from models.TableProcessModel_v2 import TableProcessModel
from Workers import ModelLoadWorker, ImageProcessWorker
from ui.logger import logger
from ui.path_select_dialog import PathSelectDialog
from tools.Utils import *
from config import load_config

app_config = load_config("config.yaml")
import pandas as pd


class TableAnalysisTool(ImageProcessApp):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("TableAnalysisTool")
        self.setWindowIcon(QIcon(self.icons_dir + "app_64x64.png"))
        self.model: TableProcessModel = None
        self.thread_ = None
        self.worker = None
        self.load_model()
        self.action = None
        # self.tool_bar.addSeparator()
        # associate_action = QAction(
        #     QIcon(self.icons_dir + "/associate.png"), "关联人员信息", self
        # )
        # self.person_info_select_dialog = PathSelectDialog(self)
        # associate_action.triggered.connect(self.person_info_select_dialog.show)
        # self.person_info_select_dialog.path_selected.connect(self.associate_person_info)
        # self.tool_bar.addAction(associate_action)

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
        self.thread_ = QThread()  # QThread: Destroyed while thread is still running
        self.worker.moveToThread(self.thread_)

        # clean old connect
        # 在连接新信号前断开旧连接

        try:
            # self.worker.show_next_signal.disconnect()
            # self.worker.finished.disconnect()
            pass
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
        except Exception as e:
            logger.error(f"locate fail, {e}", exc_info=True, stack_info=True)
            self.set_status_button_state("处理")

    def structure_table(self):
        super().structure_table()
        try:
            self.setup_action("structure")
            self.thread_.start()
            # 更新按钮状态
        except Exception as e:
            self.set_status_button_state("处理")
            logger.error(f"structure fail {e}", exc_info=True, stack_info=True)

    def recognize_table(self):
        super().recognize_table()
        try:
            self.setup_action("ocr")
            self.thread_.start()
            # 更新按钮状态
        except Exception as e:
            self.set_status_button_state("处理")
            logger.error(f"recognize fail, {e}", exc_info=True, stack_info=True)

    def process_a4(self):
        super().process_a4()
        try:
            self.setup_action("a4_eval")
            self.thread_.start()
            # 更新按钮状态
        except Exception as e:

            if self.check_images_loaded(info=False):
                self.set_status_button_state("处理")
            else:
                self.set_status_button_state("准备")
            logger.error(f"a4_eval fail, {e}", exc_info=True, stack_info=True)

    def split_a3(self):
        super().split_a3()
        try:
            self.setup_action("a3_split")
            self.thread_.start()
            # 更新按钮状态
        except Exception as e:
            if self.check_images_loaded(info=False):
                self.set_status_button_state("处理")
            else:
                self.set_status_button_state("准备")
            logger.error(f"a3_split fail, {e}", exc_info=True, stack_info=True)
            

    def process_a3(self):
        try:
            super().process_a3()
            self.setup_action("a3_eval")
            self.thread_.start()
            # 更新按钮状态
        except Exception as e:
            if self.check_images_loaded(info=False):
                self.set_status_button_state("处理")
            else:
                self.set_status_button_state("准备")
            
            logger.error(f"a3_eval fail, {e}", exc_info=True, stack_info=True)
            

    def process_a3_back(self):
        try:
            super().process_a3()
            self.setup_action("a3_eval_back")
            self.thread_.start()
            # 更新按钮状态
        except Exception as e:
            if self.check_images_loaded(info=False):
                self.set_status_button_state("处理")
            else:
                self.set_status_button_state("准备")
            logger.error(f"a3_eval_back fail, {e}", exc_info=True, stack_info=True)
            

    def export_and_open_excel(self):
        super().export_and_open_excel()
        # 默认保存 提供路径打开文件夹即可
        open_folder(app_config["app_dir"]["base_output_dir"])

    def export_and_open_history(self):
        super().export_and_open_history()
        # 默认保存 提供路径打开文件夹即可
        open_folder()
        # 如果没有保存 if
        self.model.a4table_score_eval_service.score_history_to_xlsx()


    def associate_person_info(self, person_and_back_work_dir: dict):
        """
        正面自动关联
        关联信息 A3_LEFT_NO_1 A3_RIGHT_NO_1 A3_RIGHT_NO_2 A3_RIGHT_NO_3
        person_info.xlsx
        反面 需要提供 person_info.xlsx 路径 和 反面 工作文件夹路径 才能关联
        """
        reply = QMessageBox.question(
            self,
            "重要提示",
            "请检查导出人员姓名等重要信息是否有误，如有问题取消操作，手动修改后再关联",
            QMessageBox.Ok | QMessageBox.Cancel,
        )
        if reply == QMessageBox.Cancel:
            return
        person_info_xlsx = person_and_back_work_dir["xlsx"]
        back_working_dir = person_and_back_work_dir["folder"]
        associate_back = person_and_back_work_dir["associate_back"]

        # 正/反面 都可更新
        self.model.associate(
            person_info_xlsx, back_working_dir, associate_back=associate_back
        )

        QMessageBox.information(self, "info", "关联完成")

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

        if self.action in ["a3_eval", "a4_eval", "a3_eval_back"]:
            # 导出统计
            self.model.a4table_score_eval_service.score_history_to_xlsx()
            self.model.a4table_score_eval_service.action_score_hisory_to_xlsx()
            if self.action == "a3_eval": # 正面 export person info and associate
                self.model.export_peroson_info()
                self.model.person_infos.clear()
            # 清理模型统计得分历史
            self.model.a4table_score_eval_service.score_history.clear()
            self.model.a4table_score_eval_service.action_xlsx_history.clear()
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
