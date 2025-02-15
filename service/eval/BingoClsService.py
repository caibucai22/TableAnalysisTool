# -*- coding: UTF-8 -*-
"""
@File    ：TableProcess.py
@Author  ：Csy
@Date    ：2025/2/2 19:10 
@Bref    : 返回 list[dict{bingo:,conf:}]
@Ref     :
TODO     :
         :
"""

from PIL import Image
import numpy as np
from Settings import *
from service.interface import IClsService
import numpy as np
from models.ModelManager_v2 import model_manger
from tools.Utils import img_load_by_Image, images_convert
from tools.Logger import get_logger
from ultralytics.engine.results import Boxes, Results
import logging

logger = get_logger(__file__, log_level=logging.INFO)


class BingoClsService(IClsService):
    def __init__(self):
        self.model = model_manger.get_model("bingo_cls")

    def binary_cls(self, cell_images: list) -> list:
        bingo_ret = []
        cell_images = images_convert(cell_images, to_type="Image")
        for cell in cell_images:
            bingo = {}
            model_output = self.model(cell)
            top5_conf = model_output[0]  # 1 , 0
            top5_index = model_output[1]  # 0 , 1 -> bingo , no-bingo
            bingo = {
                "bingo": top5_index[0] == 0,  # 假设索引0表示"bingo"
                "conf": top5_conf[0],  # 取最高置信度
            }
            bingo_ret.append(bingo)
        return bingo_ret


class BingoClsServiceV2(IClsService):
    """
    使用 yolo v8训练的bingo_det 模型 做分类任务 
    调用逻辑 和 yoloLocate基本一致 进行了简化 没有单独封装 BingoDetModel
    """

    def __init__(self):
        self.model = model_manger.get_model("bingo_det")

    def binary_cls(self, cell_images: list) -> list:
        bingo_ret = []
        cell_images = images_convert(cell_images, to_type="Image")
        for cell in cell_images:
            bingo = {}
            model_output: Results = self.model.predict(cell)  # auto device
            boxs = model_output[0].boxes.xyxy.cpu().numpy()
            conf = model_output[0].boxes.conf.cpu().numpy()
            if len(boxs) >= 1:
                bingo = {
                    "bingo": True,
                    "conf": conf[0],
                }
            else:
                bingo = {
                    "bingo": False,
                    "conf": 0,
                }
            bingo_ret.append(bingo)
        return bingo_ret
