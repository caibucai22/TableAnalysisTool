# -*- coding: UTF-8 -*-
"""
@File    ：TableProcess.py
@Author  ：Csy
@Date    ：2024/12/23 19:10 
@Bref    : 
@Ref     :
TODO     :
         :
"""
from ModelManager import ModelManager
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import datetime


class ScoreEvaluation():
    def __init__(self) -> None:
        self.text_rec_model = ModelManager.get_text_rec_model()
        self.bingo_judge_model = ModelManager.get_bingo_cls_model()
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
        result = self.bingo_judge_model(image)
        top5_conf = result[0]  # 1 , 0
        top5_index = result[1]  # 0 , 1 -> bingo , no-bingo

        return top5_index[0] == 0

    def eval_line_score(self, line_score_boxs):
        n_ = len(line_score_boxs)
        line_rec_ret = []
        line_rec_confidence = []
        line_bingo_state = [False] * n_
        for i, box in enumerate(line_score_boxs):
            score_box = self.cur_image.crop(box)
            is_bingo = self.judge_bingo(score_box)
            line_bingo_state[i] = is_bingo
            if line_bingo_state[i]:
                line_rec_ret.append('bingo')
                continue

            ret = self.text_rec_model.ocr(cv2.cvtColor(
                np.asarray(score_box), cv2.COLOR_RGB2BGR))
            line_rec_ret.append(ret[0][0][1][0] if ret[0]
                                is not None else 'bingo')
            line_rec_confidence.append(
                ret[0][0][1][1] if ret[0] is not None else 0)

        bingo_idx = line_rec_ret.index('bingo')  # first bingo

        increased = True
        try:
            if bingo_idx == len(line_rec_ret)-1 and bingo_idx-2 >= 0:
                increased = int(line_rec_ret[bingo_idx-2]
                                ) < int(line_rec_ret[bingo_idx-1])
            elif bingo_idx == 0 and bingo_idx+2 < len(line_rec_ret):
                increased = int(line_rec_ret[bingo_idx+1]
                                ) < int(line_rec_ret[bingo_idx+2])
            elif bingo_idx-1 >= 0 and bingo_idx+1 < len(line_rec_ret):
                increased = int(line_rec_ret[bingo_idx-1]
                                ) < int(line_rec_ret[bingo_idx+1])
        except Exception as e:
            print('run error, first judge increased failed', e)
            # second method
            judge_increased_list = []
            for i, cell_ret in enumerate(line_rec_ret):
                if str.isdigit(cell_ret):
                    judge_increased_list.append(int(cell_ret))
                    if len(judge_increased_list) == 2:
                        break
            increased = (judge_increased_list[0]-judge_increased_list[1]) < 0

        bingo_number = -1
        if bingo_idx == len(line_rec_ret)-1:
            if increased:
                bingo_number = int(line_rec_ret[bingo_idx-1])+1
            else:
                bingo_number = int(line_rec_ret[bingo_idx-1])-1
        elif bingo_idx == 0:
            if increased:
                bingo_number = int(line_rec_ret[bingo_idx+1])-1
            else:
                bingo_number = int(line_rec_ret[bingo_idx+1])+1
        else:
            if increased:
                bingo_number = int(line_rec_ret[bingo_idx-1])+1
            else:
                bingo_number = int(line_rec_ret[bingo_idx-1])-1
        return bingo_number
        # return line_rec_confidence

    def eval_score(self):
        for row_i in range(self.n_row):
            if row_i == 0:
                continue
            score_boxs = self.cells[row_i*self.n_col +
                                    self.score_col_start_idx:row_i*self.n_col+self.score_col_end_idx+1]
            line_score = self.eval_line_score(score_boxs)
            self.row_scores.append(line_score)
        self.score_history.append(
            (f'{self.cur_image_name}_score.xlsx', sum(self.row_scores)))

    def to_xlsx(self, to_stdout=False):
        if to_stdout:
            for row_i, row_score in enumerate(self.row_scores):
                print(f'row {row_i+2} ---> score: {row_score}')
        xlsx = pd.DataFrame(
            {'no': range(1, len(self.row_scores)+1), 'score': self.row_scores})
        xlsx.to_excel(
            f'{self.save_dir}/{self.cur_image_name}_score.xlsx', index=False)

    def score_history_to_xlsx(self):
        scores_collect_xlsx = pd.DataFrame({'文件名': [x for x, _ in self.score_history],
                                       "总分": [y for _, y in self.score_history]})
        cur_time = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")

        scores_collect_xlsx.to_excel(f'socres_collected_{cur_time}.xlsx',index=False)

    def load_next(self, image: Image.Image, cells, image_name, save_dir,
                  n_row, n_col, score_col_start_idx, score_col_end_idx
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
        self.score_col_start_idx = score_col_start_idx
        self.score_col_end_idx = score_col_end_idx
