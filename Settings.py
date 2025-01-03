# -*- coding: UTF-8 -*-
"""
@File    ：settings.py
@Author  ：Csy
@Date    ：2024/12/12 17:16 
@Bref    :
@Ref     :
TODO     :
         :
"""
import os
import torch

# system
CACHE = True
ENABLE_CUT_CELLS = False

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ['HF_HUB_CACHE'] = './hf_models/'
os.environ['TRANSFORMERS_CACHE'] = './hf_models'
os.environ['HF_HOME'] = './hf_models'

# paddle
PADDLE_OCR_TABLE_MODEL_DIR = '.paddleocr/whl/table/ch_ppstructure_mobile_v2.0_SLANet_infer'
PADDLE_OCR_DET_MODEL_DIR = '.paddleocr/whl/det/ch/ch_PP-OCRv4_det_infer'
PADDLE_OCR_REC_MODEL_DIR = '.paddleocr/whl/rec/ch/ch_PP-OCRv4_rec_infer'
PADDLE_OCR_LAYOUT_MODEL_DIR = ''

# yolo
BINGO_CLS_MODEL_PATH = './hf_models/bingo-cls.onnx'


# cuda
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# support image
IMAGE_EXTS = ['.jpg', '.bmp', '.png']

# support cuda
USE_DEVICE = "cuda:0" if torch.cuda.is_available() else 'cpu'
# if only cpu
# USE_DEVICE = 'cpu'

# font
FONT_PATH = './resources/simfang.ttf'

# TABLE layout related
SCORE_COL_START_IDX = 2
SCORE_COL_END_IDX = 5
