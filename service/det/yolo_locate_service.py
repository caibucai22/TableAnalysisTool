from service.interface import ITableLocateService
from models.TableLocateModels import YoloLocate
import numpy as np


class YoloLocateService(ITableLocateService):
    def __init__(self):
        self.model = YoloLocate()  # 原有定位模型

    def locate_table(self, image_path: str) -> np.array:
        bboxs,roi_imgs = self.model.infer(image_path)
        return bboxs,roi_imgs
