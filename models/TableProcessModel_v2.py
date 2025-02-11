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

from Settings import *

from PIL import Image
import numpy as np

import logging
from tools.Logger import get_logger

logger = get_logger(__file__, log_level=logging.INFO)

from models.ImageProcessModel import ImageProcessModel
from service.registry import ServiceRegistry, registry
from service.interface import ITableLocateService, IOCRService, ITableStructureService
from service.structure.cyclecenter_net_service import CycleCenterNetService
from service.custom.ScoreEvaluation_v2 import A4ScoreEvaluation
from Preprocess import A3Split
from tools.Utils import draw_locate, draw_table, img_convert_to_bgr

from config import load_config

app_config = load_config("config.yaml")
table_config = load_config("service/table_config.yaml")


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


class TableProcessModel(ImageProcessModel):
    def __init__(self, service_registry: ServiceRegistry) -> None:
        super().__init__()
        self.registry = service_registry
        # service
        self.a4table_score_eval_service: A4ScoreEvaluation = None
        self.table_locate_service: ITableLocateService = None
        self.table_structure_service: ITableStructureService = None
        self.table_rec_service: IOCRService = None
        self.table_data = None
        self.setup_services()
        self.a3_left_imgs = []
        self.a3_right_imgs = []

    def setup_services(self):
        self.table_locate_service = self.registry.get(ITableLocateService, "paddle")
        self.table_structure_service = self.registry.get(
            ITableStructureService, "table_transfromer"
        )
        self.table_rec_service = self.registry.get(IOCRService, "paddle")
        self.a4table_score_eval_service = A4ScoreEvaluation()
        logger.info("all services setuped")

    def reset_results(self):
        self.table_data = None

    def locate(self):
        bboxs, roi_imgs = self.table_locate_service.locate_table(self.image_path)  # bgr
        logger.info(f"found {len(bboxs)} tables")

    def structure(self, image_path: str):
        logger.info(f"start to split table structure")
        parsed_table = self.table_structure_service.recognize_structure()
        logger.info(f"parse table structure done and adapting")

    def ocr(self, imag_path=None) -> list:
        pass

    def a4_eval(self):
        # try:
        #     logger.info("start to parse total image")
        #     bboxs, roi_imgs = self.table_locate_service.locate_table(
        #         self.image_path
        #     )  # bgr
        #     logger.info(f"found {len(bboxs)} tables")
        # except Exception as e:
        #     logger.error(f"run error {e}")
        # for test score_eval_v2

        yolo_locate_service = registry.get(ITableLocateService, "yolo")
        cyclecenter_structure_servie: CycleCenterNetService = registry.get(
            ITableStructureService, "cyclecenter_net"
        )
        bboxs, roi_imgs = yolo_locate_service.locate_table(self.image_path)
        for i, bbox in enumerate(bboxs):
            table_image = self.cur_image.crop(bbox)
            # for i, table_image in enumerate(roi_imgs):
            logger.info(f"start to parse no.{i+1} table")
            # table_image = Image.fromarray(cv2.cvtColor(table_image, cv2.COLOR_BGR2RGB))
            logger.info(f"start to split table structure and adapt")
            self.table_data = self.table_structure_service.recognize_structure(
                table_image
            )
            # self.table_data = cyclecenter_structure_servie.recognize_structure(
            #     img_convert_to_bgr(np.array(table_image))
            # )
            if (
                app_config["app_run"]["enable_cache"]
                or app_config["app_debug"]["cache"]
            ):
                draw_table(
                    table_image.copy(),
                    self.table_data,
                    save_dir=self.each_image_cache_dir,
                    cut_cell=app_config["app_debug"]["cut_table_cells"],
                    draw_cell=True,
                )
            # assert table sructure
            # if not self.table_data.is_matched(): # gt
            #     raise Exception("structure donot matched")
            logger.info(f"split table structure done")

            logger.info("start eval score...")
            n_row, n_col = len(self.table_data.row_bbox_list), len(
                self.table_data.col_bbox_list
            )
            self.a4table_score_eval_service.load_next(
                table_image,
                self.table_data.cell_bbox_list,
                image_name=self.cur_image_name,
                image_type="A3_RIGHT_NO_3_TABLE",
                save_dir=self.each_image_cache_dir,
                n_row=n_row,
                n_col=n_col,
            )
            try:
                self.a4table_score_eval_service.eval_score()
                self.a4table_score_eval_service.to_xlsx()
            except Exception as e:
                logger.error(f"{e}")
                self.a4table_score_eval_service.eval_score(process_failure=True)
                self.a4table_score_eval_service.to_xlsx(process_failure=True)
        logger.info("current image, all table parsed")
        TableProcessModel.clear()

    def a3_split(self):
        left_path, right_path = A3Split.split(
            self.image_path, self.original_img_shape, save_dir=self.each_image_cache_dir
        )
        self.a3_left_imgs.append(left_path)
        self.a3_right_imgs.append(right_path)

    def a3_eval(self):
        # 切换到yolo_locate 先定位 左右图 然后定位 locate_type
        yolo_locate_service = registry.get(ITableLocateService, "yolo")
        cyclecenter_structure_servie: CycleCenterNetService = registry.get(
            ITableStructureService, "cyclecenter_net"
        )
        gt_left_table_structure = [table_config["A3"]["left_no_1_table"]]
        gt_rght_table_structure = [
            table_config["A3"]["right_no_1_table"],
            table_config["A3"]["right_no_2_table"],
            table_config["A3"]["right_no_3_table"],
            table_config["A3"]["right_no_4_table"],
        ]
        for left_img_path in self.a3_left_imgs:
            self.load_image(left_img_path)
            try:
                logger.info("start to parse left image")
                bboxs, roi_imgs = yolo_locate_service.locate_table(
                    self.image_path
                )  # bgr
                logger.info(f"found {len(bboxs)} tables")
            except Exception as e:
                logger.error(f"run error {e}")

            assert len(bboxs) == 1  # only 1 table
            for i, bbox in enumerate(bboxs):
                table_image_l = self.cur_image.crop(bbox)
                logger.info(f"start to parse no.{i+1} table")
                # table_image = Image.fromarray(
                #     cv2.cvtColor(table_image, cv2.COLOR_BGR2RGB)
                # )
                logger.info(f"start to split table structure and adapt")
                self.table_data = self.table_structure_service.recognize_structure(
                    table_image_l
                )
                # assert table sructure
                eval_type = gt_left_table_structure[i]["eval_type"]
                gt_cols = gt_left_table_structure[i]["cols"]
                gt_rows = gt_left_table_structure[i]["rows"]

                if eval_type == "line":
                    if not self.table_data.is_matched(gt_rows, gt_cols):  # gt
                        raise Exception("structure donot matched gt")
                logger.info(f"split table structure done")

                logger.info("start eval score...")
                n_row, n_col = len(self.table_data.row_bbox_list), len(
                    self.table_data.col_bbox_list
                )
                self.a4table_score_eval_service.load_next(
                    table_image_l,
                    self.table_data.cell_bbox_list,
                    image_name=self.cur_image_name,
                    image_type="A3_LEFT_NO_1_TABLE",
                    save_dir=self.each_image_cache_dir,
                    n_row=n_row,
                    n_col=n_col,
                    # start_col_idx=start_col_idx,
                    # end_col_idx=end_col_idx,
                )
                self.a4table_score_eval_service.eval_score()
                self.a4table_score_eval_service.to_xlsx()

        for rght_img_path in self.a3_right_imgs:
            self.load_image(rght_img_path)
            try:
                logger.info("start to parse left image")
                bboxs, roi_imgs = yolo_locate_service.locate_table(
                    self.image_path
                )  # bgr
                logger.info(f"found {len(bboxs)} tables")
            except Exception as e:
                logger.error(f"run error {e}")
            assert len(bboxs) == 4
            located_table_Images = draw_locate(
                self.image_path,
                bboxs=bboxs,
                save_dir=self.each_image_cache_dir,
                cut=True,
                return_crops=True,  # should be eabled with cut var
            )
            assert len(located_table_Images) == 4
            locate_table_type = [
                "A3_RIGHT_NO_1_TABLE",
                "A3_RIGHT_NO_2_TABLE",
                "A3_RIGHT_NO_3_TABLE",
                "A3_RIGHT_NO_4_TABLE",
            ]
            for i, (table_img_r, table_type_r) in enumerate(
                zip(located_table_Images, locate_table_type)
            ):
                logger.info(f"start to parse no.{i+1} table on right page")
                logger.info(f"start to split table structure and adapt")
                if i == 3:
                    self.table_data = cyclecenter_structure_servie.recognize_structure(
                        img_convert_to_bgr(np.array(table_img_r))
                    )
                else:
                    self.table_data = self.table_structure_service.recognize_structure(
                        table_img_r
                    )
                # assert table sructure
                eval_type = gt_rght_table_structure[i]["eval_type"]
                gt_cols = gt_rght_table_structure[i]["cols"]
                gt_rows = gt_rght_table_structure[i]["rows"]
                if eval_type == "line":
                    if not self.table_data.is_matched(gt_rows, gt_cols):  # gt
                        raise Exception("structure donot matched gt")
                logger.info(f"split table structure done")

                logger.info("start eval score...")
                # update table structure
                n_row, n_col = len(self.table_data.row_bbox_list), len(
                    self.table_data.col_bbox_list
                )

                if i < 3:
                    start_col_idx = gt_rght_table_structure[i]["eval"]["start_col_idx"]
                    end_col_idx = gt_rght_table_structure[i]["eval"]["end_col_idx"]
                    self.a4table_score_eval_service.load_next(
                        table_img_r,
                        self.table_data.cell_bbox_list,
                        image_name=self.cur_image_name,
                        image_type=table_type_r,
                        save_dir=self.each_image_cache_dir,
                        n_row=n_row,
                        n_col=n_col,
                        # start_col_idx=start_col_idx,
                        # end_col_idx=end_col_idx,
                    )
                    self.a4table_score_eval_service.eval_score()
                    self.a4table_score_eval_service.to_xlsx()
                elif i == 3:  # table 4 只传递需要 rec 的 cell
                    gt_cell_idxs = gt_rght_table_structure[i]["eval"][
                        "gt_cell_idxs"
                    ]  # 经测试方能确定的超参数
                    gt_names = gt_rght_table_structure[i]["eval"]["gt_names"]
                    rec_dict = {}
                    for i, (name, idx) in enumerate(zip(gt_names, gt_cell_idxs)):
                        eval_cell = table_img_r.crop(
                            self.table_data.cell_bbox_list[idx]
                        )
                        cell_rec_ret = self.a4table_score_eval_service.rec_single_cell(
                            eval_cell
                        )
                        if cell_rec_ret["state"]:
                            rec_dict[name] = cell_rec_ret["txt"]
                        else:  # no-recon also thinked as bingo
                            rec_dict[name] = "no rec"
                    logger.info(f"parsed {i+1} table info")
                    for key, value in rec_dict.items():
                        logger.info(f"{key}: {value}")
        logger.info("current eval_a3 epoch done!")

    def run(self, next_image_path, action=""):
        try:
            self.reset_results()
            self.load_image(next_image_path)
            if action == "a4_eval":
                self.a4_eval()
            elif action == "a3_eval":
                self.a3_split()
                self.a3_eval()
            elif action == "locate":
                self.locate()
            elif action == "structure":
                self.structure()
            elif action == "ocr":
                self.ocr()
            else:
                logger.error(f"{action} is not supported", exc_info=True)
                raise Exception(f"not supported action {action}")
        except Exception as e:
            logger.error(f"{e}")

    @staticmethod
    def clear():
        torch.cuda.empty_cache()
        device.cuda.empty_cache()


def test_a3_direct_rec_by_cell():
    pass


if __name__ == "__main__":
    # table_img_path = "C:/Users/001/Pictures/ocr/v2/v2_test_left.jpg"
    # table_img_path = "C:/Users/001/Pictures/ocr/v2/locate_table_3.jpg"
    table_img_path = "C:/Users/001/Pictures/ocr/v2/a3_single_test2.jpg"
    image_dir = "./test_images"
    table_img_path_list = [
        image_dir + "/" + imgname
        for imgname in os.listdir(image_dir)
        if os.path.splitext(imgname)[-1] in [".jpg"]
    ]

    t_class_init_start = time.time()
    table_process = TableProcessModel(service_registry=registry)
    print("model construct elapsed time ", time.time() - t_class_init_start)

    # single_test ~3.5s
    t_single_start = time.time()
    table_process.run(table_img_path, action="a3_eval")
    print("single test elapsed time ", time.time() - t_single_start)

    # multi_test ~3s
    # n = len(table_img_path_list)
    # print("found {} images".format(n))
    # t_multi_test = time.time()
    # for img_path in table_img_path_list:
    #     table_process.run(img_path, action="a4_eval")
    # print(
    #     "multi test elapsed time ",
    #     time.time() - t_multi_test,
    #     "mean time: ",
    #     (time.time() - t_multi_test) / n,
    # )
