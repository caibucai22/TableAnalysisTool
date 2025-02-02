# -*- coding: UTF-8 -*-
"""
@File    ：TableProcess.py
@Author  ：Csy
@Date    ：2024/12/23 19:10 
@Bref    : 当前逻辑 用户测试可以达到 ~98.5% 准确识别
@Ref     :
TODO     :
         :
"""
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import datetime
import os
from Settings import *

from service.rec.paddle_ocr_service import PaddleOCRService
from service.eval.BingoClsService import BingoClsService
from service.interface import IClsService, IOCRService
from adapters.Table import TableEntity

from tools.Logger import get_logger

import logging

logger = get_logger(__file__, log_level=logging.INFO)


class A4ScoreEvaluation:
    def __init__(self) -> None:
        self.ocr_service: IOCRService = PaddleOCRService()
        self.bingo_cls_service: IClsService = BingoClsService()
        self.cur_image = None
        self.cur_image_name = ""
        self.save_dir = ""
        self.cells = None
        self.n_row = -1
        self.n_col = -1
        self.score_col_start_idx = -1
        self.score_col_end_idx = -1
        self.row_scores = []
        self.score_history = []

    def judge_bingo(self, image: Image.Image) -> bool:
        bingo_ret_list = self.bingo_cls_service.binary_cls([image])
        return bingo_ret_list[0]["bingo"]

    @staticmethod
    def recover_score_list(part_list: list, n: int, line_confidence: list, **kwargs):
        def direct_parse(idx, num):
            score_list_ = []
            if (
                (idx == 0 and num == 4)
                or (idx == 1 and num == 3)
                or (idx == 2 and num == 2)
                or (idx == 3 and num == 1)
            ):
                score_list_ = [num for num in range(4, 0, -1)]
            elif (
                (idx == 0 and num == 1)
                or (idx == 1 and num == 2)
                or (idx == 2 and num == 3)
                or (idx == 3 and num == 4)
            ):
                score_list_ = [num for num in range(1, 4 + 1)]
            logger.info(
                "row_{} recover score list: {}".format(
                    kwargs.get("row_i", "i"), score_list_
                )
            )
            return score_list_

        if len(part_list) == 1:
            idx = part_list[0][0]
            num = part_list[0][1]
            if num not in [1, 2, 3, 4]:  # fix error reco ocr
                num = int(str(num)[0])
            return direct_parse(idx, num)

        score_list = [-1] * n
        increased = (part_list[0][1] - part_list[1][1]) < 0
        (idx1, num1) = part_list[0]
        (idx2, num2) = part_list[1]
        score_list[idx1] = num1
        score_list[idx2] = num2
        try:
            num1_start = num1
            num1_start2 = num1
            num2_start = num2
            # 填充第一个区间
            for i in range(idx1, -1, -1):
                score_list[i] = num1_start
                num1_start = num1_start - 1 if increased else num1_start + 1

            # 填充中间区间
            for i in range(idx1, idx2 + 1):
                score_list[i] = num1_start2
                num1_start2 = num1_start2 + 1 if increased else num1_start2 - 1
            assert score_list[idx2] == num2  # 检查边界值是否一致

            # 填充最后一个区间
            for i in range(idx2, n):
                score_list[i] = num2_start
                num2_start = num2_start + 1 if increased else num2_start - 1
            logger.info(
                "row_{} recover score list {} ".format(
                    kwargs.get("row_i", "i"), score_list
                )
            )
        except Exception as e:
            logger.warning(
                f"run error, first recover score list failed, ---> {e}, retry"
            )
            # based on confidence
            idx1_conf = line_confidence[idx1]
            idx2_conf = line_confidence[idx2]
            if idx2_conf > idx1_conf:
                idx1 = idx2
                num1 = num2

            if (
                (idx1 == 0 and num1 == 4)
                or (idx1 == 1 and num1 == 3)
                or (idx1 == 2 and num1 == 2)
                or (idx1 == 3 and num1 == 4)
            ):
                score_list = [num for num in range(4, 0, -1)]
            elif (
                (idx1 == 0 and num1 == 1)
                or (idx1 == 1 and num1 == 2)
                or (idx1 == 2 and num1 == 3)
                or (idx1 == 3 and num1 == 4)
            ):
                score_list = [num for num in range(1, 4 + 1)]
            logger.info(
                "row_{} recover score list {} ".format(
                    kwargs.get("row_i", "i"), score_list
                )
            )
        return score_list

    def rec_single_cell(self, score_box, **kwargs):
        ret_list = self.ocr_service.recognize_text([score_box])
        # ret = self.text_rec_model.ocr(
        #     cv2.cvtColor(np.asarray(score_box), cv2.COLOR_RGB2BGR), cls=False
        # )
        return ret_list[0]

    def eval_line_score(self, line_score_boxs, **kwargs):
        n_ = len(line_score_boxs)
        line_rec_ret = []
        line_rec_confidence = []
        line_bingo_state = [False] * n_
        line_success = True
        for i, box in enumerate(line_score_boxs):
            score_box = self.cur_image.crop(box)
            is_bingo = self.judge_bingo(score_box)
            line_bingo_state[i] = is_bingo
            if line_bingo_state[i]:
                line_rec_ret.append("bingo")
                line_rec_confidence.append(0)
                continue

            cell_rec_ret = self.rec_single_cell(score_box, **kwargs)
            # ret = self.text_rec_model.ocr(
            #     cv2.cvtColor(np.asarray(score_box), cv2.COLOR_RGB2BGR), cls=False
            # )
            if cell_rec_ret["state"]:
                line_rec_ret.append(cell_rec_ret["txt"])
                line_rec_confidence.append(cell_rec_ret["conf"])
            else:  # no-recon also thinked as bingo
                line_rec_ret.append("bingo")
                line_bingo_state[i] = False
                line_rec_confidence.append(0)

        bingo_idx = line_rec_ret.index("bingo")  # first bingo
        if not line_bingo_state[bingo_idx]:  # if invalid
            bingo_idx = line_rec_ret.index("bingo", bingo_idx + 1)  # second bingo
        # TODO no bingo fix

        increased = True
        judge_increased_list = []  # record first number idx and first number
        try:
            if bingo_idx == len(line_rec_ret) - 1 and bingo_idx - 2 >= 0:
                increased = int(line_rec_ret[bingo_idx - 2]) < int(
                    line_rec_ret[bingo_idx - 1]
                )
            elif bingo_idx == 0 and bingo_idx + 2 < len(line_rec_ret):
                increased = int(line_rec_ret[bingo_idx + 1]) < int(
                    line_rec_ret[bingo_idx + 2]
                )
            elif bingo_idx - 1 >= 0 and bingo_idx + 1 < len(line_rec_ret):
                increased = int(line_rec_ret[bingo_idx - 1]) < int(
                    line_rec_ret[bingo_idx + 1]
                )
        except Exception as e:
            logger.warning(f"run error, first judge increased failed ! ---> {e}")
        finally:
            # second method
            for i, cell_ret in enumerate(line_rec_ret):
                if str.isdigit(cell_ret):
                    judge_increased_list.append((i, int(cell_ret)))
                    if len(judge_increased_list) == 2:
                        break
                elif cell_ret == "一":  # fix 1 recon because of rotation
                    judge_increased_list.append((i, 1))
                    if len(judge_increased_list) == 2:
                        break
            # TODO only 1 number
            assert len(judge_increased_list) >= 1
            # increased = (judge_increased_list[0]
            #              [1]-judge_increased_list[1][0]) < 0
            # print('parse increased success')

        # bingo_number = -1
        # recover choice socres list []
        try:
            scores_list = A4ScoreEvaluation.recover_score_list(
                judge_increased_list, n_, line_rec_confidence, **kwargs
            )
        except Exception as e:
            logger.warning(
                f"recover score list failed, prepare to skip current line, ---> {e}",
            )
            line_success = False

        # if bingo_idx == len(line_rec_ret)-1:
        #     if increased:
        #         bingo_number = int(line_rec_ret[bingo_idx-1])+1
        #     else:
        #         bingo_number = int(line_rec_ret[bingo_idx-1])-1
        # elif bingo_idx == 0:
        #     if increased:
        #         bingo_number = int(line_rec_ret[bingo_idx+1])-1
        #     else:
        #         bingo_number = int(line_rec_ret[bingo_idx+1])+1
        # else:
        #     if increased:
        #         bingo_number = int(line_rec_ret[bingo_idx-1])+1
        #     else:
        #         bingo_number = int(line_rec_ret[bingo_idx-1])-1
        # return bingo_number
        # return line_rec_confidence

        # if recover_score_list error, error NoneType' object is not subscriptable
        if line_success:
            return line_success, scores_list[bingo_idx]
        return line_success, 0  # line parse failed return 0

    def eval_score(self, process_failure=False):
        if process_failure:
            self.score_history.append(
                (f"{self.cur_image_name}_score.xlsx", 0, False, "all")
            )
            return
        zero_indices = []
        for row_i in range(self.n_row):
        # for row_i in [14, 17]: #TODO: re-ocr for faile lines
            if row_i == 0:
                continue
            score_boxs = self.cells[
                row_i * self.n_col
                + self.score_col_start_idx : row_i * self.n_col
                + self.score_col_end_idx
                + 1
            ]
            try:
                line_success, line_score = self.eval_line_score(score_boxs, row_i=row_i)
                self.row_scores.append((row_i, line_score))
            except:
                logger.info("recording no. and skipping... ")
                zero_indices.append(row_i)
                self.row_scores.append((row_i, 0))

        logger.info(f"total {len(zero_indices)} lines failed --> {zero_indices}")
        self.score_history.append(
            (
                f"{self.cur_image_name}_score.xlsx",
                sum(score for no, score in self.row_scores),
                True if len(zero_indices) == 0 else False,
                ",".join(map(str, zero_indices)),
            )
        )

    def to_xlsx(self, process_failure=False, to_stdout=False):
        if process_failure:
            xlsx = pd.DataFrame(
                {
                    "no": [row_i + 1 for row_i in range(self.n_row)],
                    "score": [0 for i in range(self.n_row)],
                }
            )
            xlsx.to_excel(
                f"{self.save_dir}/{self.cur_image_name}_score.xlsx", index=False
            )
            return
        if to_stdout:
            for row_i, row_score in enumerate(self.row_scores):
                logger.info(f"row {row_i+2} ---> score: {row_score}")
        xlsx = pd.DataFrame(
            {
                "no": [x for x, _ in self.row_scores],
                "score": [y for _, y in self.row_scores],
            }
        )
        xlsx.to_excel(f"{self.save_dir}/{self.cur_image_name}_score.xlsx", index=False)

    def score_history_to_xlsx(self):
        scores_collect_xlsx = pd.DataFrame(
            {
                "文件名": [x for x, _, _, _ in self.score_history],
                "总分": [y for _, y, _, _ in self.score_history],
                "状态": [
                    "success" if z else "uncompleted"
                    for _, _, z, _ in self.score_history
                ],
                "未识别": [unrecon for _, _, _, unrecon in self.score_history],
            }
        )
        cur_time = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")

        if SAVE_TO_USER_HOME:
            # 构建保存路径并解析用户主目录
            save_path = os.path.expanduser("~/Documents/scores_collected")
            directory = os.path.join(save_path, f"scores_collected_{cur_time}.xlsx")
            # 确保目录存在
            os.makedirs(os.path.dirname(directory), exist_ok=True)
            # 保存为 Excel 文件
            scores_collect_xlsx.to_excel(directory, index=False)
        else:
            scores_collect_xlsx.to_excel(
                f"socres_collected_{cur_time}.xlsx", index=False
            )

    def load_next(
        self,
        image: Image.Image,
        cells,
        image_name,
        image_type,
        save_dir,
        n_row,
        n_col,
    ):
        # clear
        self.row_scores.clear()
        # prepare
        self.cur_image = image
        self.cells = cells
        self.cur_image_name = image_name
        self.save_dir = save_dir
        self.n_row = n_row
        self.n_col = n_col
        self.table_type = image_type

        if self.table_type == "A4_SINGLE_TABLE":
            self.score_col_start_idx = 2
            self.score_col_end_idx = 5
        elif self.table_type == "A4_LEFT_SINGLE_TABLE":
            self.score_col_start_idx = 1
            self.score_col_end_idx = 1
        elif self.table_type == "A4_RIGHT_NO_1_TABLE":
            self.score_col_start_idx = 1
            self.score_col_end_idx = 1
        elif self.table_type == "A4_RIGHT_NO_2_TABLE":
            self.score_col_start_idx = 1
            self.score_col_end_idx = 1
        elif self.table_type == "A4_RIGHT_NO_3_TABLE":
            self.score_col_start_idx = 1
            self.score_col_end_idx = 1
        elif self.table_type == "A4_RIGHT_NO_4_TABLE":
            self.score_cell_names = []
            self.score_gt_cells = []
        else:
            raise Exception("")
