# -*- coding: UTF-8 -*-
"""
@File    ：TableProcess.py
@Author  ：Csy
@Date    ：2024/12/12 17:10 
@Bref    : cpu ~ 3s cuda ~ 2s ; finally cuda ~3.3s ; no-cache ~ 3s
@Ref     :
TODO     :
         :
"""
import os
import time
import torch
from paddle import device
from Settings import *

from typing import Union
from PIL import Image

import logging
from tools.Logger import get_logger

logger = get_logger(__file__, log_level=logging.INFO)

from config import load_config

app_config = load_config("config.yaml")

# print(app_config)


def timeit_decorator(enable_print=True):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            if enable_print:
                print(
                    f"{func.__name__} executed in {end_time - start_time:.6f} seconds"
                )
            return result

        return wrapper

    return decorator


class ImageProcessModel:
    def __init__(
        self,
    ) -> None:
        self.image_path = ""
        self.original_img_shape = None
        self.cur_image = None
        self.cur_image_name = ""
        self.cur_image_dir = (
            ""
            if app_config["app_dir"]["base_output_dir"] is None
            else app_config["app_dir"]["base_output_dir"]
        )  # base dir
        self.cache_base_dir = (
            ""
            if app_config["app_dir"]["sub_cache_dir"] is None
            else app_config["app_dir"]["sub_cache_dir"]
        )  # cache dir
        self.cache_dir = ""
        self.each_image_cache_dir = ""

    @timeit_decorator(enable_print=False)
    def load_image(self, image_path):
        self.image_path = image_path
        self.cur_image = Image.open(image_path).convert("RGB")  # RGB
        self.original_img_shape = self.cur_image.size
        self.cur_image_dir = (
            os.path.dirname(self.image_path)
            if self.cur_image_dir == ""
            else self.cur_image_dir
        )
        self.cur_image_name = os.path.basename(self.image_path).split(".")[0]
        self.cache_dir = self.cur_image_dir + "/" + self.cache_base_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.initialize_cache_dir()
        logger.info(f"base output dir {self.cur_image_dir}")
        logger.info(f"sub cache dir {self.cache_dir}")
        logger.info(f"current img cache dir {self.each_image_cache_dir}")

    def initialize_cache_dir(self):
        self.each_image_cache_dir = self.cache_dir + "/" + self.cur_image_name
        os.makedirs(self.each_image_cache_dir, exist_ok=True)

    def reset_results(self):
        pass

    def process(self):
        pass

    @staticmethod
    def clear():
        torch.cuda.empty_cache()
        device.cuda.empty_cache()


if __name__ == "__main__":
    table_img_path = "C:/Users/001/Pictures/ocr/v2/v2_test_left.jpg"
    image_dir = "./test_images"
    table_img_path_list = [
        image_dir + "/" + imgname
        for imgname in os.listdir(image_dir)
        if os.path.splitext(imgname)[-1] in [".jpg"]
    ]
    app = ImageProcessModel()
    app.run(table_img_path)
