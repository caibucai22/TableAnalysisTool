from transformers import (
    TableTransformerForObjectDetection,
    DetrFeatureExtractor,
)
from paddleocr import PPStructure, PaddleOCR
from models.YoloClsInfer import Yolov8_cls_PIL
from models.WiredTableRecognition import WiredTableStructureModel
from Settings import *

import logging
from tools.Logger import get_logger
logger = get_logger(__file__, log_level=logging.INFO)


class ModelManager:
    _table_locate_model = None
    _table_structure_feature_extractor_model = None
    _table_structure_split_model = None
    _text_rec_model = None
    _bingo_cls_model = None
    _table_wired_structure_split_model = None

    @staticmethod
    def get_table_wired_structure_split_model():
        if ModelManager._table_wired_structure_split_model is None:
            logger.info("loading wired table structure split model")
            ModelManager._table_wired_structure_split_model = WiredTableStructureModel()
        return ModelManager._table_wired_structure_split_model

    @staticmethod
    def get_bingo_cls_model():
        if ModelManager._bingo_cls_model is None:
            logger.info("Loading Bingo Judge Model...")
            ModelManager._bingo_cls_model = Yolov8_cls_PIL(
                model_path=BINGO_CLS_MODEL_PATH
            )
        return ModelManager._bingo_cls_model

    @staticmethod
    def get_text_rec_model():
        if ModelManager._text_rec_model is None:
            logger.info("Loading Text Rec Model...")
            ModelManager._text_rec_model = PaddleOCR(
                sho_log=False,
                det_model_dir=PADDLE_OCR_DET_MODEL_DIR,
                rec_model_dir=PADDLE_OCR_REC_MODEL_DIR,
                use_gpu=True if USE_DEVICE == "cuda:0" else False,
            )
        return ModelManager._text_rec_model

    @staticmethod
    def get_table_locate_model():
        if ModelManager._table_locate_model is None:
            logger.info("Loading Table Locate Model...")
            ModelManager._table_locate_model = PPStructure(
                show_log=False,
                table_model_dir=PADDLE_OCR_TABLE_MODEL_DIR,
                det_model_dir=PADDLE_OCR_DET_MODEL_DIR,
                rec_model_dir=PADDLE_OCR_REC_MODEL_DIR,
                use_gpu=True if USE_DEVICE == "cuda:0" else False,
            )
        return ModelManager._table_locate_model

    @staticmethod
    def get_table_structure_feature_extractor_model() -> DetrFeatureExtractor:
        if ModelManager._table_structure_feature_extractor_model is None:
            logger.info("Loading Table Structure Feature Extractor Model...")
            ModelManager._table_structure_feature_extractor_model = (
                DetrFeatureExtractor()
            )
        return ModelManager._table_structure_feature_extractor_model

    @staticmethod
    def get_table_structure_split_model():
        if ModelManager._table_structure_split_model is None:
            logger.info("Loading Table Structure Split Model...")
            ModelManager._table_structure_split_model = TableTransformerForObjectDetection.from_pretrained(
                "microsoft/table-transformer-structure-recognition-v1.1-all",
                # 'hf_models/models--microsoft--table-transformer-structure-recognition-v1.1-all/snapshots/7587a7ef111d9dcbf8ac695f1376ab7014340a0c', # for local model
                cache_dir="./hf_models",  # debug path ./hf_models
                proxies={
                    "http": "http://127.0.0.1:7890",
                    "https": "http://127.0.0.1:7890",
                },
                device_map=(
                    "cuda:0" if USE_DEVICE == "cuda:0" else "cpu"
                ),  # need accelerate > 0.26.0
            )
        return ModelManager._table_structure_split_model
