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
