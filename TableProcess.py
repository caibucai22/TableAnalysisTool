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
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from ModelManager import ModelManager
from ScoreEvaluation import ScoreEvaluation
from Settings import *


# def timeit_decorator(enable_print=True):
#     def decorator(func):
#         def wrapper(*args, **kwargs):
#             start_time = time.time()
#             result = func(*args, **kwargs)
#             end_time = time.time()
#             if enable_print:
#                 print(f"{func.__name__} executed in {end_time - start_time:.6f} seconds")
#             return result
#         return wrapper
#
#     return decorator


def timeit_decorator(enable_print=True):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            if enable_print:
                print(
                    f"{func.__name__} executed in {end_time - start_time:.6f} seconds")
            return result

        return wrapper

    return decorator


class TableProcessModel:
    def __init__(self) -> None:
        self.image_path = ""

        self.cur_image = None
        self.cur_image_name = ""
        self.cur_image_dir = ""  # base dir
        self.cache_dir = ""  # cache dir
        self.each_image_mid_dir = ""

        # 引用模型管理器中的模型
        self.table_locate_model = ModelManager.get_table_locate_model()
        self.table_structure_feature_extractor_model = ModelManager.get_table_structure_feature_extractor_model()
        self.table_structure_split_model = ModelManager.get_table_structure_split_model()
        self.score_eval = ScoreEvaluation()

        self.table_locate_result = {}
        self.table_split_result = {}

        self.locate_table_bbox = []

        self.cells_box_list = []
        self.cols_box_list = []
        self.rows_box_list = []

    @timeit_decorator(enable_print=False)
    def load_image(self):
        self.cur_image = Image.open(self.image_path).convert("RGB")  # RGB
        self.cur_image_dir = os.path.dirname(self.image_path)
        self.cur_image_name = os.path.basename(self.image_path).split('.')[0]

    def initialize_cache_dir(self):
        self.cache_dir = self.cur_image_dir + '/' + 'cache'
        os.makedirs(self.cache_dir, exist_ok=True)
        self.each_image_mid_dir = os.path.join(
            self.cache_dir, self.cur_image_name)

    def reset_results(self):
        """重置所有推理结果相关变量"""
        self.table_locate_result = {}
        self.table_split_result = {}
        self.locate_table_bbox = []
        self.cells_box_list = []
        self.cols_box_list = []
        self.rows_box_list = []

    @timeit_decorator(enable_print=False)
    def infer_locate_table(self):  # ~2s
        # result = self.table_locate_model(self.cur_image) # 使用的是cv2 读取 这里是PIL Image 需要转换
        # result = self.table_locate_model(
        #     cv2.cvtColor(np.asarray(self.cur_image), cv2.COLOR_RGB2BGR)) # rgb2bgr
        # support ch path ch image name
        img = cv2.imdecode(np.fromfile(
            self.image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"无法读取图片：{img_path}")
        result = self.table_locate_model(img)  # bgr
        # pillow image -> cv2 和 直接 cv2 检测结果有差距很大

        os.makedirs(self.each_image_mid_dir, exist_ok=True)
        for region in result:
            if region['type'] == 'table':
                roi_img = region['img']
                if CACHE:
                    img_path = os.path.join(self.each_image_mid_dir,
                                            '{}_{}.jpg'.format(self.cur_image_name, "located_table"))
                    # cv2.imwrite(img_path, roi_img)
                    cv2.imencode('.jpg', roi_img)[1].tofile(img_path)
                self.locate_table_bbox = region['bbox']
                return roi_img
                # break

    @timeit_decorator(enable_print=False)
    def encoding_for_table_split(self, image: Image):
        encoding = self.table_structure_feature_extractor_model(
            image, return_tensors="pt")
        self.table_split_result["encoding"] = encoding
        return encoding

    @timeit_decorator(enable_print=False)
    def infer_split(self, encoding, target_sizes):
        if encoding['pixel_values'].device == torch.device('cpu') and USE_DEVICE == 'cuda:0':
            encoding['pixel_values'] = encoding['pixel_values'].cuda()
            encoding['pixel_mask'] = encoding['pixel_mask'].cuda()
        with torch.no_grad():
            outputs = self.table_structure_split_model(**encoding)
        results = self.table_structure_feature_extractor_model.post_process_object_detection(
            outputs, threshold=0.85, target_sizes=target_sizes)[0]  # fix threshold=0.85 hyper-parameters

        self.table_split_result.update(results)
        del encoding

    @timeit_decorator(enable_print=False)
    def parse_table_split_result(self):
        '''
        {0: 'table', 1: 'table column', 2: 'table row', 3: 'table column header',
        4: 'table projected row header', 5: 'table spanning cell'}
        '''
        self.cols_box_list = [self.table_split_result['boxes'][i].tolist() for i in range(len(
            self.table_split_result['boxes'])) if self.table_split_result['labels'][i].item() == 1]
        # TODO cols_nms
        self.rows_box_list = [self.table_split_result['boxes'][i].tolist() for i in range(len(
            self.table_split_result['boxes'])) if self.table_split_result['labels'][i].item() == 2]

        self.cell_match_table()

    def cell_match_table(self):
        self.rows_box_list = sorted(
            self.rows_box_list, key=lambda x: ((x[2]+x[0])/2 + (x[3]+x[1])/2))
        self.cols_box_list = sorted(
            self.cols_box_list, key=lambda x: ((x[2]+x[0])/2 + (x[3]+x[1])/2))
        self.cells_box_list = [TableProcessModel.intersection(row, col) for row in self.rows_box_list for col in
                               self.cols_box_list]

    def setup_score_eval(self, image_score):
        n_row, n_col = len(self.rows_box_list), len(self.cols_box_list)
        self.score_eval.load_next(image_score, self.cells_box_list,
                                  image_name=self.cur_image_name, save_dir=self.cur_image_dir,
                                  n_row=n_row, n_col=n_col,
                                  score_col_start_idx=SCORE_COL_START_IDX,
                                  score_col_end_idx=SCORE_COL_END_IDX
                                  )

    @staticmethod
    def intersection(boxA: list, boxB: list):
        """
        计算两个 Box 的交集
        Args:
            boxA: 第一个 Box，(x1, y1, x2, y2)
            boxB: 第二个 Box，(x1, y1, x2, y2)

        Returns:
            如果两个 Box 有交集，返回交集的 Box，否则返回 None
        """

        # 计算交集的左上角和右下角坐标
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # 如果交集的宽度或高度小于等于0，则说明两个 Box 没有交集
        intersection_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        return [xA, yA, xB, yB] if intersection_area > 0 else None

    @timeit_decorator(enable_print=False)
    def draw_boxs(self, image: Image, draw_col=True, draw_row=True, draw_cell=True, cut_cell=False):
        if draw_col and len(self.cols_box_list) > 0:
            col_draw_image = image.copy()
            for col in self.cols_box_list:
                col_draw = ImageDraw.Draw(col_draw_image)
                col_draw.rectangle(col, outline="red", width=3)
            col_draw_image.save(self.each_image_mid_dir + '/' + 'cols.jpg')

        if draw_row and len(self.rows_box_list) > 0:
            row_draw_image = image.copy()
            for row in self.rows_box_list:
                row_draw = ImageDraw.Draw(row_draw_image)
                row_draw.rectangle(row, outline="red", width=3)
            row_draw_image.save(self.each_image_mid_dir + '/' + 'rows.jpg')

        if draw_cell and len(self.cells_box_list) > 0:
            cell_draw_image = image.copy()
            for cell in self.cells_box_list:
                cell_draw = ImageDraw.Draw(cell_draw_image)
                cell_draw.rectangle(cell, outline="red", width=3)
            cell_draw_image.save(self.each_image_mid_dir + '/' + 'cells.jpg')

        if draw_cell and (len(self.cells_box_list) > 0):
            white_background = Image.new(
                "RGB", (image.width, image.height), (255, 255, 255))
            font = ImageFont.truetype(FONT_PATH, size=20)
            sorted_cell_draw = ImageDraw.Draw(white_background)
            for i, cell in enumerate(self.cells_box_list):
                sorted_cell_draw.rectangle(cell, outline='red', width=3)
                sorted_cell_draw.text(xy=(cell[0]+(cell[2]-cell[0])/2, cell[1]+(cell[3]-cell[1])/2),
                                      text=str(i), font=font, fill=(0, 0, 255))
                sorted_cell_draw.text(xy=(cell[0]+(cell[2]-cell[0])/2-20, cell[1]+(cell[3]-cell[1])/2-20),
                                      text=str(cell[0])+"_"+str(cell[1]), font=font, fill=(0, 255, 0))

            white_background.save(
                self.each_image_mid_dir + '/' + 'sorted_cells.jpg')
        if cut_cell and (len(self.cells_box_list) > 0):
            print('enable cutting cells')
            for i, cell in enumerate(self.cells_box_list):
                x = i // len(self.cols_box_list)
                y = i % len(self.cols_box_list)

                cell_image = image.crop(cell)
                cell_image.save(self.each_image_mid_dir +
                                '/' + f'cell_{x}_{y}' + '.jpg')
            print('cutting cells done')

    @timeit_decorator(enable_print=False)
    def run_parse_table(self):
        table_image = self.infer_locate_table()  # bgr
        if len(self.locate_table_bbox) == 0:
            raise Exception("定位表格失败")

        # table_image = self.cur_image.crop(self.locate_table_bbox) # RGB
        table_image = Image.fromarray(
            cv2.cvtColor(table_image, cv2.COLOR_BGR2RGB))

        # table_image = Image.open(
        #     self.each_image_mid_dir + '/' + '{}_{}.jpg'.format(self.cur_image_name, "located_table")).convert("RGB")
        target_sizes = [table_image.size[::-1]]

        self.encoding_for_table_split(table_image)
        if self.table_split_result['encoding'] is None:
            raise Exception("表格特征编码失败")
        self.infer_split(self.table_split_result['encoding'], target_sizes)
        if len(self.table_split_result.keys()) <= 1:
            raise Exception("表格切分失败")
        self.parse_table_split_result()
        # visualize first for debug
        if CACHE:
            self.draw_boxs(table_image.copy(), cut_cell=ENABLE_CUT_CELLS)

        self.setup_score_eval(table_image)

    def run(self, next_image_path):
        try:
            self.reset_results()
            self.image_path = next_image_path
            self.load_image()
            self.initialize_cache_dir()
            self.run_parse_table()
            # assert table sructure
            if len(self.cols_box_list) != 6:
                raise Exception('structure donot matched')
            self.score_eval.eval_score()
            self.score_eval.to_xlsx()
        except Exception as e:
            print('run error ', e)
            self.score_eval.eval_score(process_failure=True)
            self.score_eval.to_xlsx(process_failure=True)

    @staticmethod
    def clear():
        torch.cuda.empty_cache()
        device.cuda.empty_cache()


if __name__ == "__main__":
    table_img_path = 'C:/Users/001/Pictures/ocr/1_6_1544/微信图片_20250106154529.jpg'
    image_dir = './test_images'
    table_img_path_list = [image_dir + '/' + imgname for imgname in os.listdir(image_dir) if
                           os.path.splitext(imgname)[-1] in ['.jpg']]

    t_class_init_start = time.time()
    table_process = TableProcessModel()
    print('model construct elapsed time ', time.time() - t_class_init_start)

    # single_test ~3.5s
    t_single_start = time.time()
    table_process.run(table_img_path)
    print('single test elapsed time ', time.time() - t_single_start)

    # multi_test ~3s
    # n = len(table_img_path_list)
    # print("found {} images".format(n))
    # t_multi_test = time.time()
    # for img_path in table_img_path_list:
    #     table_process.run(img_path)
    # print('multi test elapsed time ', time.time() - t_multi_test, 'mean time: ',
    #       (time.time() - t_multi_test) / n)

    table_process.clear()
