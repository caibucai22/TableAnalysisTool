from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
import time

# from models.TableProcessModel import TableProcessModel
from models.TableProcessModel_v2 import TableProcessModel
from service.registry import registry
import logging
from tools.Logger import get_logger

logger = get_logger(__file__, log_level=logging.INFO)


class ModelLoadWorker(QObject):
    model_loaded = pyqtSignal(object)
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.model = None

    def run(self):
        try:
            self.model = TableProcessModel(service_registry=registry)
            self.model_loaded.emit(self.model)
        except Exception as e:
            logger.error(f"loading model {e}")
        finally:
            self.finished.emit()


class ImageProcessWorker(QObject):
    # one_image_processed = pyqtSignal(str)
    all_image_processed = pyqtSignal(str)
    finished = pyqtSignal()
    show_next_signal = pyqtSignal(int)

    def __init__(self, images, model: TableProcessModel, action=None, log=False):
        super().__init__()
        self.images = images
        self.processor = model
        self.action = action
        self.log = log

    @pyqtSlot()
    def run(self):
        try:
            for i, image_path in enumerate(self.images):
                # 处理图片
                if self.log:
                    logger.info(f"processing {image_path}")
                self.processor.run(image_path, action=self.action)
                if self.log:
                    logger.info(f"process {image_path} done")
                time.sleep(0.5)
                self.show_next_signal.emit(i)
                # self.one_image_processed.emit(f"Processed: {image_path}")
            self.all_image_processed.emit("all done!")
        except Exception as e:
            self.one_image_processed.emit(f"Error processing {image_path}: {str(e)}")
            logger.error(f"Worker process failed {image_path}", exc_info=True)
        finally:
            self.finished.emit()  # 处理完成 finally keep finished signal must emit
