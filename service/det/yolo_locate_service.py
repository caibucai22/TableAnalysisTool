from service.interface import ITableLocateService
from models.TableLocateModels import YoloLocate
import numpy as np


class YoloLocateService(ITableLocateService):
    def __init__(self):
        self.model = YoloLocate()  # 原有定位模型

    def locate_table(self, image_path: str) -> np.array:
        boxs = self.model.infer(image_path)
        return boxs
