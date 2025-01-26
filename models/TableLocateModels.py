# -*- coding: UTF-8 -*-
"""
@File    ：app.py
@Author  ：Csy
@Date    ：2024/1/22 16:58 
@Bref    : 检测定位图中的表格 返回坐标xyxy
@Ref     :
TODO     :
         :
"""
import os
from typing import Union
from ultralyticsplus import YOLO, render_result
from ultralytics.engine.results import Boxes, Results
from models.ModelManager import ModelManager
import cv2
import numpy as np
from Settings import *
import logging
from Logger import get_logger

logger = get_logger(__file__, log_level=logging.INFO)

import Utils


class PaddleLocate:

    def __init__(self) -> None:
        """
        返回 表格bbox 和 表格roi图像
        """
        self.model = ModelManager.get_table_locate_model()

    def infer_locate_table(self, img, **kwargs) -> np.ndarray:
        base_image_name = kwargs.get("basename", "img")
        save_dir = kwargs.get("save_dir", ".")
        result = self.model(img)  # bgr # TODO: 定位+识别 识别重复,不进行识别
        """
        result list[tuple]
        tuple type: title table table_caption reference
        tuple bbox
        tuple img
        tuple res
        """
        bboxs = []
        roi_imgs = []
        for region in result:
            if region["type"] == "table":
                roi_img = region["img"]
                if CACHE:
                    img_path = os.path.join(
                        save_dir,
                        "{}_{}.jpg".format(base_image_name, "located_table"),
                    )
                    cv2.imencode(".jpg", roi_img)[1].tofile(img_path)

                roi_imgs.append(roi_img)
                bboxs.append(region["bbox"])

        return bboxs, roi_imgs

    def infer(self, img: Union[str], **kwargs):
        if isinstance(img, str):
            save_dir, img_name = os.path.split(img)
            img_basename, _ = os.path.splitext(img_name)
            img = Utils.img_load_by_cv2(img)

        try:
            bboxs, roi_imgs = self.infer_locate_table(
                img, save_dir=save_dir, basename=img_basename
            )
            for i in range(len(bboxs)):
                logger.info(f"no.{i+1} xyxy:{bboxs[i]}")
        except Exception as e:
            logger.error("locate table infer failed", exc_info=True)
            return [], []
        return bboxs, roi_imgs


class YoloLocate:
    def __init__(self) -> None:
        """
        返回 bbox, 支持多表
        """
        self.model = YOLO(
            "E:/my-github-repos/01-my_repos/TableAnalysisTool/hf_models/yolov8m_table_detection.pt"
        )
        # set model parameters
        self.model.overrides["conf"] = 0.55  # NMS confidence threshold
        self.model.overrides["iou"] = 0.45  # NMS IoU threshold
        self.model.overrides["agnostic_nms"] = False  # NMS class-agnostic
        self.model.overrides["max_det"] = 100  # maximum number of detections per image

    def infer_(self, img: Union[str]):
        if isinstance(img, str):
            img = Utils.img_load_by_Image(img).convert("RGB")

        results = self.model.predict(img)
        logger.info("YoloLocate inferring")
        return results

    def parse(self, model_output: Results):
        cls = model_output[0].boxes.cls.cpu()
        conf = model_output[0].boxes.conf.cpu()
        names: dict = model_output[0].names
        boxs = model_output[0].boxes.xyxy.cpu()
        n = boxs.shape[0]
        for i in range(n):
            logger.info(
                f"no.{i+1} xyxy:{boxs[i]} type:{names[int(cls[i])]},conf:{conf[i]}"
            )
        return boxs

    def infer(self, img: Union[str]):
        return self.parse(self.infer_(img=img))


def test_yoloLocate():
    model = YoloLocate()
    img_path = "C:/Users/001/Pictures/ocr/v2/right.jpg"
    save_dir, image_name = os.path.split(img_path)
    image_basename, _ = os.path.splitext(image_name)
    bboxs = model.infer(img=img_path)
    Utils.draw_locate(img_path, bboxs=bboxs, save_dir=save_dir, cut=True)


def test_paddleLocate():
    model = PaddleLocate()
    img_path = "C:/Users/001/Pictures/ocr/v2/right.jpg"
    save_dir, image_name = os.path.split(img_path)
    bboxs, roi_imgs = model.infer(img_path)
    print(f"total {len(bboxs)} tables")
    Utils.draw_locate(
        img_path, bboxs=bboxs, save_dir=save_dir, cut=True, adjust_ratio=0.0
    )


if __name__ == "__main__":
    # test_yoloLocate()
    test_paddleLocate()
