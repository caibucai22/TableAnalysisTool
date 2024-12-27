from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
import time
from TableProcess import TableProcessModel


class ModelLoadWorker(QObject):
    model_loaded = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.model = None

    def run(self):
        try:
            self.model = TableProcessModel()
        except Exception as e:
            print('error loading model', e)
        else:
            self.model_loaded.emit(self.model)


class ImageProcessWorker(QObject):
    image_processed = pyqtSignal(str)
    finished = pyqtSignal()
    show_signal = pyqtSignal(int)

    def __init__(self, images, model:TableProcessModel,log=False):
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
                    print('processing ', image_path, '--->', end='')
                self.processor.run(image_path)
                if self.log:
                    print('done')
                time.sleep(0.5)
                self.image_processed.emit(f"Processed: {image_path}")

            except Exception as e:
                self.image_processed.emit(
                    f"Error processing {image_path}: {str(e)}")

        self.finished.emit()  # 处理完成
