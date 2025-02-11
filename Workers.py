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

    def __init__(self):
        super().__init__()
        self.model = None

    def run(self):
        try:
            self.model = TableProcessModel(service_registry=registry)
        except Exception as e:
            logger.error(f"loading model {e}")
        else:
            self.model_loaded.emit(self.model)


class ImageProcessWorker(QObject):
    image_processed = pyqtSignal(str)
    finished = pyqtSignal()
    show_signal = pyqtSignal(int)

    def __init__(self, images, model: TableProcessModel, log=False):
        super().__init__()
        self.images = images
        self.processor = model
        self.log = log

    @pyqtSlot()
    def run(self):
        for i, image_path in enumerate(self.images):
            try:
                self.show_signal.emit(i)
                # 处理图片
                if self.log:
                    logger.info(f"processing {image_path}")
                self.processor.run(image_path,action='a4_eval')
                if self.log:
                    logger.info(f"process {image_path} done")
                time.sleep(0.5)
                self.image_processed.emit(f"Processed: {image_path}")

            except Exception as e:
                self.image_processed.emit(f"Error processing {image_path}: {str(e)}")

        self.finished.emit()  # 处理完成
