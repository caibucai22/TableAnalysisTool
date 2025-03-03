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
import os, glob
import time
import torch
from paddle import device
import cv2

from Settings import *

from PIL import Image
import numpy as np
import pandas as pd

import logging, datetime
from tools.Logger import get_logger

logger = get_logger(__file__, log_level=logging.INFO)

from models.ImageProcessModel import ImageProcessModel
from service.registry import ServiceRegistry, registry
from service.interface import ITableLocateService, IOCRService, ITableStructureService
from service.structure.cyclecenter_net_service import CycleCenterNetService
from service.custom.ScoreEvaluation_v2 import A4ScoreEvaluation
from Preprocess import A3Split
from tools.Utils import draw_locate, draw_table, img_convert_to_bgr, adjst_box

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
        self.person_infos = dict()
        self.person_info_path = None

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

    def a4_eval(self, image_type="A4_SINGLE_TABLE"):
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
            cycle_tabledata = cyclecenter_structure_servie.recognize_structure(
                img_convert_to_bgr(np.array(table_image)),
                table_type=image_type
            )

            if (
                app_config["app_run"]["enable_cache"]
                or app_config["app_debug"]["cache"]
                or app_config["app_debug"]["debug"]
            ):
                draw_table(
                    table_image.copy(),
                    self.table_data,
                    save_dir=self.each_image_cache_dir,
                    cut_cell=app_config["app_debug"]["cut_table_cells"],
                    draw_cell=app_config["app_debug"]["draw_cell"],
                    prefix=image_type,
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
                image_type=image_type,
                save_dir=self.each_image_cache_dir,
                n_row=n_row,
                n_col=n_col,
                cycle_tabledata=cycle_tabledata,
            )
            try:
                self.a4table_score_eval_service.eval_score()
                self.a4table_score_eval_service.to_xlsx()
            except Exception as e:
                logger.error(f"{e}", exc_info=True, stack_info=True)
                self.a4table_score_eval_service.eval_score(process_failure=True)
                self.a4table_score_eval_service.to_xlsx(process_failure=True)
        logger.info("current image, all table parsed")
        TableProcessModel.clear()

    def a3_split(self):
        self.a3_left_imgs.clear()
        self.a3_right_imgs.clear()
        left_path, right_path = A3Split.split(
            self.image_path, self.original_img_shape, save_dir=self.each_image_cache_dir
        )
        self.a3_left_imgs.append(left_path)
        self.a3_right_imgs.append(right_path)

    def a3_eval_back(self):
        locate_table_type = [
            "A3_BACK_NO_2_TABLE",
            "A3_BACK_NO_3_TABLE",
        ]
        back_images = [self.a3_left_imgs[-1], self.a3_right_imgs[-1]]
        for i, img_path in enumerate(back_images):
            self.load_image(img_path)
            self.a4_eval(image_type=locate_table_type[i])

    def a3_eval_left(self):
        yolo_locate_service = registry.get(ITableLocateService, "yolo")
        cyclecenter_structure_servie: CycleCenterNetService = registry.get(
            ITableStructureService, "cyclecenter_net"
        )
        gt_left_table_structure = [table_config["A3"]["left_no_1_table"]]
        for left_img_path in self.a3_left_imgs:
            self.load_image(left_img_path)
            try:
                logger.info("start to parse left image")
                bboxs, roi_imgs = yolo_locate_service.locate_table(
                    self.image_path
                )  # bgr
                logger.info(f"found {len(bboxs)} tables")
            except Exception as e:
                logger.error(f"run error {e}", exc_info=True, stack_info=True)

            assert len(bboxs) == 1  # only 1 table
            for i, bbox in enumerate(bboxs):
                table_image_l = self.cur_image.crop(bbox)
                logger.info(f"start to parse no.{i+1} table")
                try:
                    logger.info(f"start to split table structure and adapt")
                    self.table_data = self.table_structure_service.recognize_structure(
                        table_image_l
                    )
                    cycle_tabledata = cyclecenter_structure_servie.recognize_structure(
                        img_convert_to_bgr(np.array(table_image_l)),
                        table_type="A3_LEFT_NO_1_TABLE",
                    )
                    if (
                        app_config["app_run"]["enable_cache"]
                        or app_config["app_debug"]["cache"]
                        or app_config["app_debug"]["debug"]
                    ):
                        draw_table(
                            table_image_l.copy(),
                            self.table_data,
                            save_dir=self.each_image_cache_dir,
                            cut_cell=app_config["app_debug"]["cut_table_cells"],
                            draw_cell=app_config["app_debug"]["draw_cell"],
                            prefix="A3_LEFT_NO_1_TABLE",
                        )

                        # draw_table(
                        #     table_img_r.copy(),
                        #     cycle_tabledata,
                        #     save_dir=self.each_image_cache_dir,
                        #     cut_cell=app_config["app_debug"]["cut_table_cells"],
                        #     draw_cell=app_config["app_debug"]["draw_cell"],
                        #     prefix=table_type_r+"cycle_",
                        # )

                    # assert table sructure
                    eval_type = gt_left_table_structure[i]["eval_type"]
                    gt_cols = gt_left_table_structure[i]["cols"]
                    gt_rows = gt_left_table_structure[i]["rows"]
                    is_match_gt = True
                    if eval_type == "line":
                        if not self.table_data.is_matched(gt_rows, gt_cols):  # gt
                            logger.warning(
                                f"current col= {self.table_data.n_col} row = {self.table_data.n_row}, gt_cols = {gt_cols} gt_rows = {gt_rows}"
                            )
                            raise Exception("structure donot matched gt")
                    logger.info(f"split table structure done")
                except Exception as e:
                    is_match_gt = False
                    logger.error(
                        f"{e}", exc_info=True, stack_info=True
                    )  # just log, keep next loop

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
                    cycle_tabledata=cycle_tabledata,
                    is_match_gt=is_match_gt,
                )
                self.a4table_score_eval_service.eval_score()
                self.a4table_score_eval_service.to_xlsx()

    def a3_eval_right(self):
        yolo_locate_service = registry.get(ITableLocateService, "yolo")
        cyclecenter_structure_servie: CycleCenterNetService = registry.get(
            ITableStructureService, "cyclecenter_net"
        )
        gt_rght_table_structure = [
            table_config["A3"]["right_no_1_table"],
            table_config["A3"]["right_no_2_table"],
            table_config["A3"]["right_no_3_table"],
            table_config["A3"]["right_no_4_table"],
        ]
        for rght_img_path in self.a3_right_imgs:
            self.load_image(rght_img_path)
            try:
                logger.info("start to parse right image")
                bboxs, roi_imgs = yolo_locate_service.locate_table(
                    self.image_path
                )  # bgr
                logger.info(f"found {len(bboxs)} tables")
            except Exception as e:
                logger.error(f"run error {e}", exc_info=True, stack_info=True)
            assert len(bboxs) == 4
            locate_table_type = [
                "A3_RIGHT_NO_1_TABLE",
                "A3_RIGHT_NO_2_TABLE",
                "A3_RIGHT_NO_3_TABLE",
                "A3_RIGHT_NO_4_TABLE",
            ]
            for i, (bbox, table_type_r) in enumerate(zip(bboxs, locate_table_type)):
                # for debug
                # i = 1
                # bbox = bboxs[i]
                # table_type_r = locate_table_type[i]
                # for debug
                logger.info(f"start to parse no.{i+1} table on right page")
                logger.info(f"start to split table structure and adapt")

                table_img_r = self.cur_image.crop(adjst_box(bbox, ratio=0.1))  # TODO:
                try:
                    if i == 3:
                        self.table_data = (
                            cyclecenter_structure_servie.recognize_structure(
                                img_convert_to_bgr(np.array(table_img_r)),
                                table_type=table_type_r
                            )
                        )
                    else:
                        self.table_data = (
                            self.table_structure_service.recognize_structure(
                                table_img_r, gt_table_config=gt_rght_table_structure[i]
                            )
                        )
                        cycle_tabledata = (
                            cyclecenter_structure_servie.recognize_structure(
                                img_convert_to_bgr(np.array(table_img_r)),
                                table_type=table_type_r
                            )
                        )

                    if (
                        app_config["app_run"]["enable_cache"]
                        or app_config["app_debug"]["cache"]
                        or app_config["app_debug"]["debug"]
                    ):
                        logger.info(f"drawing table {table_type_r} sturcture")
                        draw_table(
                            table_img_r.copy(),
                            self.table_data,
                            save_dir=self.each_image_cache_dir,
                            cut_cell=app_config["app_debug"]["cut_table_cells"],
                            draw_cell=app_config["app_debug"]["draw_cell"],
                            prefix=table_type_r,
                        )
                        if i != 3:
                            draw_table(
                                table_img_r.copy(),
                                cycle_tabledata,
                                save_dir=self.each_image_cache_dir,
                                cut_cell=app_config["app_debug"]["cut_table_cells"],
                                draw_cell=app_config["app_debug"]["draw_cell"],
                                prefix=table_type_r + "_cycle_",
                            )
                    # assert table sructure
                    eval_type = gt_rght_table_structure[i]["eval_type"]
                    gt_cols = gt_rght_table_structure[i]["cols"]
                    gt_rows = gt_rght_table_structure[i]["rows"]
                    is_match_gt = True
                    if eval_type == "line":
                        if not self.table_data.is_matched(gt_rows, gt_cols):  # gt
                            draw_table(
                                table_img_r.copy(),
                                self.table_data,
                                save_dir=self.each_image_cache_dir,
                                cut_cell=app_config["app_debug"]["cut_table_cells"],
                                draw_cell=True,
                                prefix=table_type_r,
                            )
                            logger.warning(
                                f"current col= {self.table_data.n_col} row = {self.table_data.n_row}, gt_cols = {gt_cols} gt_rows = {gt_rows}"
                            )
                            raise Exception("structure donot matched gt")
                    logger.info(f"split table structure done")
                except Exception as e:
                    is_match_gt = False
                    logger.error(
                        f"{e}", exc_info=True, stack_info=True
                    )  # just log, keep next loop

                logger.info("start eval score...")
                # update table structure
                n_row, n_col = len(self.table_data.row_bbox_list), len(
                    self.table_data.col_bbox_list
                )

                if i < 3:  # score
                    try:
                        self.a4table_score_eval_service.load_next(
                            table_img_r,
                            self.table_data.cell_bbox_list,
                            image_name=self.cur_image_name,
                            image_type=table_type_r,
                            save_dir=self.each_image_cache_dir,
                            n_row=n_row,
                            n_col=n_col,
                            is_match_gt=is_match_gt,
                            cycle_tabledata=cycle_tabledata,
                        )
                        self.a4table_score_eval_service.eval_score()
                        self.a4table_score_eval_service.to_xlsx()
                    except Exception as e:
                        logger.error(
                            "eval score failed", exc_info=True, stack_info=True
                        )
                elif i == 3:  # table 4 只传递需要 rec 的 cell
                    gt_cell_idxs = gt_rght_table_structure[i]["eval"][
                        "gt_cell_idxs"
                    ]  # 经测试方能确定的超参数
                    gt_names = gt_rght_table_structure[i]["eval"]["gt_names"]
                    rec_dict = {}
                    person_info = ""
                    for cell_i, (name, idx) in enumerate(zip(gt_names, gt_cell_idxs)):
                        eval_cell = table_img_r.crop(
                            adjst_box(
                                self.table_data.cell_bbox_list[idx], enlarge=False
                            )
                        )
                        cell_rec_ret = self.a4table_score_eval_service.rec_single_cell(
                            eval_cell
                        )
                        if cell_rec_ret["state"]:
                            if name == "学籍号":  # check if not num +1 next
                                # cut_str_n =
                                cur_txt = cell_rec_ret["txt"]
                                if str.isdigit(str(cur_txt[2]).strip()):
                                    rec_dict[name] = cell_rec_ret["txt"]
                                else:
                                    logger.warning("学籍号检测错误,检测下一个cell")
                                    eval_cell = table_img_r.crop(
                                        adjst_box(
                                            self.table_data.cell_bbox_list[idx + 1],
                                            enlarge=False,
                                        )
                                    )
                                    cell_rec_ret = (
                                        self.a4table_score_eval_service.rec_single_cell(
                                            eval_cell
                                        )
                                    )
                                    rec_dict[name] = (
                                        cell_rec_ret["txt"]
                                        if cell_rec_ret["state"]
                                        else "no rec"
                                    )
                                # 限制长度
                                gt_name_len = gt_rght_table_structure[i]["eval"]["name_len"]
                                if len(rec_dict[name]) > gt_name_len:
                                    # 截断
                                    rec_dict[name] = rec_dict[name][:gt_name_len]
                            else:
                                rec_dict[name] = cell_rec_ret["txt"]

                        else:  # no-recon also thinked as bingo
                            rec_dict[name] = "no rec"
                    logger.info(f"parsed {table_type_r} table info")
                    for key, value in rec_dict.items():
                        logger.info(f"{key}: {value}")
                        person_info += value
                        # save to action_history
                        if key not in self.person_infos.keys():
                            self.person_infos[key] = []
                        self.person_infos[key].append(value)
                    # save info img
                    rollback_dir_level = "../"
                    if app_config["app_dir"]["enable_time_dir"]:
                        rollback_dir_level = "../../"
                    table_img_r.save(
                        self.each_image_cache_dir
                        + "/"
                        + rollback_dir_level
                        + person_info
                        + ".jpg"
                    )
                # for debug
                # break
                # for debug

    def a3_eval(self):
        self.a3_eval_left()
        self.a3_eval_right()
        logger.info("current eval_a3 epoch done!")

    def export_and_open_excel(self):
        pass

    def export_and_open_history(self):
        pass

    def export_peroson_info(self):
        rollback_dir_level = "../"
        if app_config["app_dir"]["enable_time_dir"]:
            rollback_dir_level = "../../"
        xlsx = pd.DataFrame(self.person_infos)
        self.person_info_path = (
            self.each_image_cache_dir + f"/{rollback_dir_level}person_info.xlsx"
        )
        xlsx.to_excel(
            self.person_info_path,
            index=False,
        )

        self.associate(
            self.person_info_path, self.cache_dir[:-6], associate_back=False
        )  # remove "cache"
        logger.info("正面表格信息已关联")

    def run(self, next_image_path, action="", cur_time=""):

        try:
            self.reset_results()
            self.load_image(next_image_path)
            # for enable_time_dir
            if app_config["app_dir"]["enable_time_dir"]:
                cur_time = cur_time + "/"
            self.cache_base_dir = cur_time + "cache"  # fixd dir
            self.cache_dir = (
                self.cur_image_dir + "/" + self.cache_base_dir
            )  # baseoutput + time_dir(if enable) + cache
            # os.makedirs(self.cache_dir) # call once in initialize_cache_dir
            # for enable_time_dir
            self.initialize_cache_dir()
            if action == "a4_eval":
                self.a4_eval()
            elif action == "a3_split":
                self.a3_split()
            elif action == "a3_eval":
                self.a3_split()
                self.a3_eval()
            elif action == "a3_eval_back":
                self.a3_split()
                self.a3_eval_back()
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
            logger.error(f"{e}", exc_info=True, stack_info=True)

    @staticmethod
    def clear():
        torch.cuda.empty_cache()
        device.cuda.empty_cache()

    def associate(
        self,
        person_info_path,
        working_dir,
        associate_back=True,
    ):
        # associate
        col_map = {
            "person_name": "姓名",
            "person_id": "学籍号",
            "problem_id": "no",
            "score": "score",
            "file_path": "文件名",
        }
        collect_table_names = []
        if associate_back:
            collect_table_names = [
                "A3_BACK_NO_2_TABLE",
                "A3_BACK_NO_3_TABLE",
            ]
        else:
            collect_table_names = [
                "A3_LEFT_NO_1_TABLE",
                "A3_RIGHT_NO_1_TABLE",
                "A3_RIGHT_NO_2_TABLE",
                "A3_RIGHT_NO_3_TABLE",
            ]
        logger.info(f"关联文件夹为：{working_dir}")
        try:
            # person_df
            person_df = pd.read_excel(
                person_info_path, usecols=[col_map["person_name"], col_map["person_id"]]
            )
            each_type_excels = []
            for prefix in collect_table_names:
                search_pattern = working_dir + f"/{prefix}*.xlsx"
                each_type_excels.extend(glob.glob(search_pattern))
            if not each_type_excels:
                raise FileNotFoundError("未找到每种表的汇总excel")

            # 处理每种表
            for i, cur_table_type_excel_path in enumerate(each_type_excels):
                table_type = os.path.basename(cur_table_type_excel_path)
                cur_person_i = 0
                one_type_collect_df = pd.read_excel(cur_table_type_excel_path)
                # 列名校验
                required_cols = [col_map["file_path"]]
                if not set(required_cols).issubset(one_type_collect_df.columns):
                    raise ValueError(f"文件 {cur_table_type_excel_path} 缺少必要列")

                same_tables = one_type_collect_df[required_cols[0]]
                same_tables_df = pd.DataFrame({})
                for same_table_path in same_tables:  # 读取同类型的 每一张解析表 并拼接
                    cur_person_id = person_df.iloc[cur_person_i][col_map["person_id"]]
                    # 拿到 题号和得分 拼接
                    same_table = pd.read_excel(
                        same_table_path,
                        usecols=[col_map["problem_id"], col_map["score"]],
                    )
                    # 拼接 图片路径 列 + 学籍号列
                    img_name = os.path.basename(same_table_path)
                    img_name = img_name[: img_name.index("half") + 4] + ".jpg"
                    same_table[col_map["file_path"]] = img_name
                    same_table[col_map["person_id"]] = cur_person_id
                    same_tables_df = pd.concat([same_tables_df, same_table])
                    cur_person_i += 1
                # 保存同一表
                # reorder
                same_tables_df = same_tables_df[
                    [
                        col_map["file_path"],
                        col_map["problem_id"],
                        col_map["score"],
                        col_map["person_id"],
                    ]
                ]
                same_tables_df.to_excel(
                    working_dir + f"/{collect_table_names[i]}.xlsx", index=False
                )

        except Exception as e:
            logger.warning("associate failed")
            logger.error(f"{e}", exc_info=True, stack_info=True)
