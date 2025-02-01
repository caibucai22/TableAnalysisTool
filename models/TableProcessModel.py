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
import cv2
from ScoreEvaluation import ScoreEvaluation
from Settings import *
import logging
from tools.Logger import get_logger
from typing import Union
from models.TableLocateModels import PaddleLocate, YoloLocate
from models.TableTransformer import TableTransformer
from models.WiredTableRecognition import WiredTableStructureModel
from adapters.TableTransformerAdapter import TableTransformerAdapter
from adapters.CycleCenterNetAdapter import CycleCenterNetAdapter
import tools.Utils as Utils
from adapters.Table import TableEntity
from PIL import Image

logger = get_logger(__file__, log_level=logging.INFO)


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


class TableProcessModel:
    def __init__(
        self,
        table_locate_model: Union[PaddleLocate, YoloLocate, None] = None,
        table_struct_model: Union[
            TableTransformer, WiredTableStructureModel, None
        ] = None,
        ocr_model=None,
    ) -> None:
        self.image_path = ""

        self.cur_image = None
        self.cur_image_name = ""
        self.cur_image_dir = ""  # base dir
        self.cache_dir = ""  # cache dir
        self.each_image_mid_dir = ""

        self.load_models(table_locate_model, table_struct_model, ocr_model)

        # service
        self.score_eval = ScoreEvaluation()
        self.table_data = None

    def load_models(self, table_locate_model, table_struct_model, ocr_model):
        if isinstance(table_locate_model, PaddleLocate) or table_locate_model is None:
            self.table_locate_model = PaddleLocate()
        else:
            logger.error("don't support current table_locate_model")

        if (
            isinstance(table_struct_model, TableTransformer)
            or table_struct_model is None
        ):
            self.table_structre_model = TableTransformer()
            self.adapter = TableTransformerAdapter()
        elif isinstance(table_struct_model, WiredTableStructureModel):
            self.table_structre_model = WiredTableStructureModel()
            self.adapter = CycleCenterNetAdapter()
        else:
            logger.error("don't support current table_struct_model")

        # self.ocr_model = ocr_model
        logger.info("all models loaded")

    @timeit_decorator(enable_print=False)
    def load_image(self, image_path):
        self.image_path = image_path
        self.cur_image = Image.open(image_path).convert("RGB")  # RGB
        self.cur_image_dir = os.path.dirname(self.image_path)
        self.cur_image_name = os.path.basename(self.image_path).split(".")[0]

    def initialize_cache_dir(self):
        self.cache_dir = self.cur_image_dir + "/" + "cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.each_image_mid_dir = os.path.join(self.cache_dir, self.cur_image_name)

    def reset_results(self):
        self.table_data = None

    def setup_score_eval(self, image_score):
        n_row, n_col = len(self.table_data.row_bbox_list), len(
            self.table_data.col_bbox_list
        )
        self.score_eval.load_next(
            image_score,
            self.table_data.cell_bbox_list,
            image_name=self.cur_image_name,
            save_dir=self.cur_image_dir,
            n_row=n_row,
            n_col=n_col,
            score_col_start_idx=SCORE_COL_START_IDX,
            score_col_end_idx=SCORE_COL_END_IDX,
        )

    @timeit_decorator(enable_print=False)
    def run_parse_table(self, table_image):
        table_image = Image.fromarray(cv2.cvtColor(table_image, cv2.COLOR_BGR2RGB))
        logger.info(f"start to split table structure")
        result_dict = self.table_structre_model.infer(table_image)
        logger.info(f"split table structure done and adapting")
        self.table_data = self.adapter.adapt(result_dict)

        # visualize first for debug
        if CACHE:
            Utils.draw_table(
                table_image.copy(),
                self.table_data,
                save_dir=self.cur_image_dir,
                cut_cell=ENABLE_CUT_CELLS,
            )
        logger.info(f"current table parsed done")
        logger.info(f"setup score evaluation service")
        self.setup_score_eval(table_image)
        return self.table_data

    def run_parse_img(self):
        logger.info("start to parse total image")
        bboxs, roi_imgs = self.table_locate_model.infer(self.image_path)  # bgr
        logger.info(f"found {len(bboxs)} tables")
        for i, table_image in enumerate(roi_imgs):
            logger.info(f"start to parse no.{i+1} table")
            self.run_parse_table(table_image)
        logger.info("current image, all table parsed")

    def run(self, next_image_path):
        try:
            self.reset_results()
            self.load_image(next_image_path)
            self.initialize_cache_dir()
            self.run_parse_img()
            # assert table sructure
            # if not self.table_data.is_matched(): # gt
            #     raise Exception("structure donot matched")
            self.score_eval.eval_score()
            self.score_eval.to_xlsx()
        except Exception as e:
            print("run error ", e)
            self.score_eval.eval_score(process_failure=True)
            self.score_eval.to_xlsx(process_failure=True)

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

    t_class_init_start = time.time()
    table_process = TableProcessModel()
    print("model construct elapsed time ", time.time() - t_class_init_start)

    # single_test ~3.5s
    t_single_start = time.time()
    table_process.run(table_img_path)
    print("single test elapsed time ", time.time() - t_single_start)

    # multi_test ~3s
    n = len(table_img_path_list)
    print("found {} images".format(n))
    t_multi_test = time.time()
    for img_path in table_img_path_list:
        table_process.run(img_path)
    print('multi test elapsed time ', time.time() - t_multi_test, 'mean time: ',
          (time.time() - t_multi_test) / n)

    table_process.clear()
