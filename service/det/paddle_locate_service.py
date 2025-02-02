from service.interface import ITableLocateService
from models.TableLocateModels import PaddleLocate
import numpy as np
from models.ModelManager_v2 import model_manger


class PaddleLocateService(ITableLocateService):
    def __init__(self):
        self.model = PaddleLocate()  # 原有定位模型

    def locate_table(self, image_path: str) -> tuple:
        bboxs, roi_imgs = self.model.infer(image_path)
        return bboxs, roi_imgs
