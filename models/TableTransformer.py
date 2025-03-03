# -*- coding: UTF-8 -*-
"""
@File    ：TableTransformer.py
@Author  ：Csy
@Date    ：2025/1/15 17:10 
@Bref    : 
@Ref     :
TODO     :
         :
"""
from models.ModelManager import ModelManager
from PIL import Image
from Settings import *
import torch
from typing import Union
from tools.Logger import get_logger
from tools import Utils

logger = get_logger(__file__)


class TableTransformer:

    def __init__(self) -> None:
        self.feature_extractor_model = (
            ModelManager.get_table_structure_feature_extractor_model()
        )

        self.structure_split_model = ModelManager.get_table_structure_split_model()

    def get_feature_encoding(self, image: Image):
        encoding = self.feature_extractor_model(image, return_tensors="pt")
        return encoding

    def infer_split(self, encoding, target_sizes):
        if (
            encoding["pixel_values"].device == torch.device("cpu")
            and USE_DEVICE == "cuda:0"
        ):
            encoding["pixel_values"] = encoding["pixel_values"].cuda()
            encoding["pixel_mask"] = encoding["pixel_mask"].cuda()
        with torch.no_grad():
            outputs = self.structure_split_model(**encoding)
        results = self.feature_extractor_model.post_process_object_detection(
            outputs, threshold=0.85, target_sizes=target_sizes
        )[
            0
        ]  # fix threshold=0.65 hyper-parameters 0.65 A3 0.85 A4
        del encoding
        return results

    def infer(self, image: Union[str, Image.Image]) -> dict:
        if isinstance(image, str):
            image = Utils.img_load_by_Image(image)
        target_sizes = [image.size[::-1]]
        try:
            encoding = self.get_feature_encoding(image)
            results = self.infer_split(encoding, target_sizes)
        except Exception as e:
            logger.error("feature encoding or infer split failed", exc_info=True)
            return {}
        return results
