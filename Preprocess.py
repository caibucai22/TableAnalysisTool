# -*- coding: UTF-8 -*-
"""
@File    ：TableProcess.py
@Author  ：Csy
@Date    ：2025/1/4 17:10 
@Bref    : 
@Ref     :
TODO     :
         :
"""

from rapid_undistorted.inference import InferenceEngine
from models.ModelManager_v2 import model_manger
from tools.Utils import img_load_by_cv2, img_wirte_by_cv2
import os.path as osp
from tools.Logger import get_logger
import logging

logger = get_logger(__file__, log_level=logging.INFO)

import cv2


class RapidUndisotrtPreprocess:

    def __init__(self, img_path: str) -> None:
        self.img_path = img_path
        self.engine = model_manger.get_model("rapid_undistort_engine_preprocess")
        self.result_img = img_load_by_cv2(img_path=img_path)
        self.original_shape = self.result_img.shape
        self.elapsed_times = {}
        self.save_path = None

    def unwarp(self):
        self.result_img, self.elapsed_times["unwarp"] = self.engine(
            self.result_img, ["unwrap"]
        )

        return self

    def unshadow(self):
        # support "binarize"
        self.result_img, self.elapsed_times["unshadow"] = self.engine(
            self.result_img, ["unshadow"]
        )
        return self

    def unblur(self):
        self.result_img, self.elapsed_times["unblur"] = self.engine(
            self.result_img, [("unblur", "OpenCvBilateral")]
        )
        return self

    def transpose(self):
        height, width, channels = self.original_shape
        if self.original_shape == self.result_img.shape:
            pass
        else:
            logger.info("transponsing ...")
            self.result_img = cv2.flip(cv2.transpose(self.result_img), 0)

    def save(self):
        self.transpose()
        image_dir, image_name = osp.split(self.img_path)
        image_basename, _ = osp.splitext(image_name)
        self.save_path = image_dir + f"/undistort_{image_basename}.jpg"
        self.new_shape = self.result_img.shape
        img_wirte_by_cv2(self.result_img, self.save_path)


class A3Split:

    def __init__(self) -> None:
        pass

    @staticmethod
    def split(img_path, original_shape):
        save_dir, image_name = osp.split(img_path)
        image_basename, _ = osp.splitext(image_name)
        img = img_load_by_cv2(img_path=img_path)

        # 获取图像尺寸
        height, width, channels = img.shape
        # print("original shape", original_shape)
        # print("undistort shape", img.shape)

        vertical = img.shape == original_shape
        half = None
        other_half = None
        if vertical:
            logger.info("vertical split")
            # 计算分割线位置
            mid_x = width // 2  # vertical
            # 分割图像
            left_half = img[:, :mid_x]
            right_half = img[:, mid_x:]

            half = left_half
            other_half = right_half
        else:
            logger.info("horizontal split")
            mid_y = height // 2  # horizontal
            up_half = img[:mid_y, :]
            down_half = img[mid_y:, :]

            half = cv2.flip(cv2.transpose(up_half), 0)
            other_half = cv2.flip(cv2.transpose(down_half), 0)
            rotated_img = cv2.flip(cv2.transpose(img), 0)
            cv2.imencode(".jpg", rotated_img)[1].tofile(
                save_dir + "/" + f"fix_{image_basename}.jpg"
            )

        # 保存分割后的图像
        cv2.imencode(".jpg", half)[1].tofile(
            save_dir + "/" + f"{image_basename}_left_half.jpg"
        )
        cv2.imencode(".jpg", other_half)[1].tofile(
            save_dir + "/" + f"{image_basename}_right_half.jpg"
        )

    @staticmethod
    def split_by_locates():
        """基于检测到的定位框切分"""
        pass


def test_rapid_undistort_prepocess():
    img_path = "C:/Users/001/Pictures/ocr/v2/微信图片_20250117213143.jpg"

    img_preprocess = RapidUndisotrtPreprocess(img_path)
    img_preprocess.unshadow().unblur().save()


def test_a3split():
    img_path = "C:/Users/001/Pictures/ocr/v2/undistort_微信图片_20250117213143.jpg"
    original_shape = img_load_by_cv2(img_path).shape
    a3split = A3Split()
    a3split.split(img_path=img_path, original_shape=original_shape)


if __name__ == "__main__":
    # test_rapid_undistort_prepocess()
    test_a3split()
