# -*- coding: UTF-8 -*-
"""
@File    ：TableProcess.py
@Author  ：Csy
@Date    ：2025/01/07 17:10 
@Bref    : 
@Ref     : wired_table_rec.main.py & support cuda by replace session object
TODO     :
         :
"""
from typing import Any, List, Union
from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel
from Settings import *
from wired_table_rec.table_line_rec import TableLineRecognition
from wired_table_rec.table_line_rec_plus import TableLineRecognitionPlus
from wired_table_rec.table_recover import TableRecover
from wired_table_rec.utils_table_recover import (
    box_4_2_poly_to_box_4_1,
    sorted_ocr_boxes,
)

import cv2
from PIL import Image, UnidentifiedImageError
import numpy as np
import traceback
import time

from pathlib import Path

InputType = Union[str, np.ndarray, bytes, Path]
from tools.Exceptions import OnnxRuntimeError
from tools.Utils import img_load_by_Image, img_convert_to_bgr

import logging
from tools.Logger import get_logger

logger = get_logger(__file__, log_level=logging.WARNING)


class OrtInferSession:

    def __init__(self, model_path, device, num_threads: int = -1) -> None:
        self.num_threads = num_threads
        exec_provider = "CPUExecutionProvider"
        provider_options = {}
        self.sess_opt = None
        if device == "cpu":
            self._init_cpu_sess_opt()
            cpu_exec_provider_options = {
                "arena_extend_strategy": "kSameAsRequested",
            }
            provider_options = cpu_exec_provider_options
        elif device == "cuda:0":
            # cuda_ep = 'CUDAExecutionProvider'
            exec_provider = "CUDAExecutionProvider"
        else:
            raise OnnxRuntimeError("not support other inference device")

        exec_provider_list = [(exec_provider, provider_options)]
        try:
            self.session = InferenceSession(
                model_path, sess_options=self.sess_opt, providers=exec_provider_list
            )
        except TypeError:
            pass

    def _init_cpu_sess_opt(self):
        self.sess_opt = SessionOptions()
        self.sess_opt.log_severity_level = 4
        self.sess_opt.enable_cpu_mem_arena = False

        if self.num_threads != -1:
            self.sess_opt.intra_op_num_threads = self.num_threads

        self.sess_opt.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    def get_input_names(self):
        input_names = []
        for node in self.session.get_inputs():
            input_names.append(node.name)
        return input_names

    def get_output_name(self, output_idx=0):
        output_names = []
        for node in self.session.get_outputs():
            output_names.append(node.name)
        return output_names[output_idx].name

    def get_metadata(self):
        meta_dict = self.session.get_modelmeta().custom_metadata_map
        return meta_dict

    def __call__(self, input_content: list):
        input_dict = dict(zip(self.get_input_names(), input_content))
        try:
            return self.session.run(None, input_feed=input_dict)
        except Exception as e:
            error_info = traceback.format_exc()
            logger.error(error_info, exc_info=True)


class WiredTableStructureModel:

    def __init__(self, table_model_path=None, version="v2") -> None:
        if version == "v2":
            model_path = (
                table_model_path
                if table_model_path
                else WIRED_TABLE_STRUCTURE_MODEL_V2_PATH
            )
            self.model = TableLineRecognitionPlus(str(model_path))
            # modify session ; support cuda
            self.model.session = OrtInferSession(model_path, device=USE_DEVICE)
        else:
            model_path = (
                table_model_path
                if table_model_path
                else WIRED_TABLE_STRUCTURE_MODEL_PATH
            )
            self.table_line_rec = TableLineRecognition(str(model_path))

        self.table_recover = TableRecover()

    def __call__(self, img, **kwargs):
        col_threshold = 15
        row_threshold = 10
        if kwargs:
            # don't need, only parse structure
            # rec_again = kwargs.get("rec_again", True)
            # need_ocr = kwargs.get("need_ocr", True)
            col_threshold = kwargs.get("col_threshold", 15)
            row_threshold = kwargs.get("row_threshold", 10)

        # preprocess + infer +  postprocess
        # !important postprocess sorted_ocr_boxes(threshold)
        polygons, rotated_polygons = self.model(img, **kwargs)

        if polygons is None:
            logger.warning("polygons is none.")
            return None, None
        try:
            table_res, logi_points = self.table_recover(
                rotated_polygons, row_threshold, col_threshold
            )
            # 将坐标由逆时针转为顺时针方向，后续处理与无线表格对齐
            polygons[:, 1, :], polygons[:, 3, :] = (
                polygons[:, 3, :].copy(),
                polygons[:, 1, :].copy(),
            )

            sorted_polygons, idx_list = sorted_ocr_boxes(
                [box_4_2_poly_to_box_4_1(box) for box in polygons]
            )
            return sorted_polygons, logi_points[idx_list]
        except:
            pass


def main():
    image_path = "./test_images/table1.jpg"
    img = img_load_by_Image(image_path)
    img = img_convert_to_bgr(np.array(img))

    model = WiredTableStructureModel(WIRED_TABLE_STRUCTURE_MODEL_V2_PATH)
    # single test
    now = time.time()
    polygons, logits = model(img=img)  # single test: CUDA 4.2 CPU 1.3
    print(f"elapse {time.time() - now}")
    print(len(polygons))

    # multi test
    image_dir = "C:/Users/001/Pictures/ocr/jpg"
    img_list = [image_dir + "/" + image for image in os.listdir(image_dir)]
    img_inputs = [
        img_convert_to_bgr(np.array(img_load_by_Image(img_input)))
        for img_input in img_list
    ]
    now = time.time()
    for img in img_inputs:
        polygons, logits = model(img=img)  # mean time 0.6s based on 4060 GPU
    print(f"elapse {time.time() - now}")
    print("mean time, ", (time.time() - now) / len(img_list))


if __name__ == "__main__":
    main()
