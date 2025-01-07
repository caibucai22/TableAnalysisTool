# -*- coding: UTF-8 -*-
"""
@File    ：TableProcess.py
@Author  ：Csy
@Date    ：2025/01/07 17:10 
@Bref    : 
@Ref     : wired_table_rec.main.py
TODO     :
         :
"""
from typing import Any, List, Union
from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel
from Exceptions import OnnxRuntimeError
from Settings import *
from wired_table_rec.table_line_rec import TableLineRecognition
from wired_table_rec.table_line_rec_plus import TableLineRecognitionPlus
from wired_table_rec.table_recover import TableRecover
from wired_table_rec.utils_table_recover import (
    box_4_2_poly_to_box_4_1,
    sorted_ocr_boxes,
)
from Logger import get_logger
import cv2
from PIL import Image, UnidentifiedImageError
import numpy as np
import traceback

logger = get_logger(__file__)

from pathlib import Path
from io import BytesIO

InputType = Union[str, np.ndarray, bytes, Path]
from Exceptions import LoadImageError


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
            raise OnnxRuntimeError("not support other inference")

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


class LoadImage:
    def __init__(
        self,
    ):
        pass

    def __call__(self, img: InputType) -> np.ndarray:
        if not isinstance(img, InputType.__args__):
            raise LoadImageError(
                f"The img type {type(img)} does not in {InputType.__args__}"
            )

        img = self.load_img(img)
        img = self.convert_img(img)
        return img

    def load_img(self, img: InputType) -> np.ndarray:
        if isinstance(img, (str, Path)):
            self.verify_exist(img)
            try:
                img = np.array(Image.open(img))
            except UnidentifiedImageError as e:
                raise LoadImageError(f"cannot identify image file {img}") from e
            return img

        if isinstance(img, bytes):
            img = np.array(Image.open(BytesIO(img)))
            return img

        if isinstance(img, np.ndarray):
            return img

        raise LoadImageError(f"{type(img)} is not supported!")

    def convert_img(self, img: np.ndarray):
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if img.ndim == 3:
            channel = img.shape[2]
            if channel == 1:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            if channel == 2:
                return self.cvt_two_to_three(img)

            if channel == 4:
                return self.cvt_four_to_three(img)

            if channel == 3:
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            raise LoadImageError(
                f"The channel({channel}) of the img is not in [1, 2, 3, 4]"
            )

        raise LoadImageError(f"The ndim({img.ndim}) of the img is not in [2, 3]")

    @staticmethod
    def cvt_four_to_three(img: np.ndarray) -> np.ndarray:
        """RGBA → BGR"""
        r, g, b, a = cv2.split(img)
        new_img = cv2.merge((b, g, r))

        not_a = cv2.bitwise_not(a)
        not_a = cv2.cvtColor(not_a, cv2.COLOR_GRAY2BGR)

        new_img = cv2.bitwise_and(new_img, new_img, mask=a)
        new_img = cv2.add(new_img, not_a)
        return new_img

    @staticmethod
    def cvt_two_to_three(img: np.ndarray) -> np.ndarray:
        """gray + alpha → BGR"""
        img_gray = img[..., 0]
        img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

        img_alpha = img[..., 1]
        not_a = cv2.bitwise_not(img_alpha)
        not_a = cv2.cvtColor(not_a, cv2.COLOR_GRAY2BGR)

        new_img = cv2.bitwise_and(img_bgr, img_bgr, mask=img_alpha)
        new_img = cv2.add(new_img, not_a)
        return new_img

    @staticmethod
    def verify_exist(file_path: Union[str, Path]):
        if not Path(file_path).exists():
            raise LoadImageError(f"{file_path} does not exist.")


class WiredTableStructureModel:

    def __init__(self, table_model_path, version="v2") -> None:
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

    def post_processing(self):
        pass


def main():
    image_path = "./test_images/table0.jpg"
    # img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    # img = cv2.cvtColor(np.array(Image.open(image_path)), cv2.COLOR_RGB2BGR)
    load_img = LoadImage()
    img = load_img.load_img(image_path)

    model = WiredTableStructureModel(WIRED_TABLE_STRUCTURE_MODEL_PATH)
    polygons, logits = model(img=img)
    print(len(polygons))
    # 图片加载不一致
    # session info 推理完不一致


if __name__ == "__main__":
    main()
