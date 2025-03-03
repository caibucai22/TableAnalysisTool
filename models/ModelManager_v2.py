import time
from typing import Dict, Any, Callable, Optional
from rapid_undistorted.inference import InferenceEngine
from models.YoloClsInfer import Yolov8_cls_PIL
from models.WiredTableRecognition import WiredTableStructureModel
from ultralyticsplus import YOLO
from transformers import (
    TableTransformerForObjectDetection,
    DetrFeatureExtractor,
)
from paddleocr import PPStructure, PaddleOCR

from Settings import *
import threading
import logging
from tools.Logger import get_logger

logger = get_logger(__file__, log_level=logging.INFO)


class ModelManager:
    """
    模型管理器，支持懒加载、线程安全、动态配置和资源释放
    在原来的基础上进行了升级 可以使用统一的 get_model register_model release_model进行管理
    """

    _table_locate_model = None
    _table_structure_feature_extractor_model = None
    _table_structure_split_model = None
    _text_rec_model = None
    _bingo_cls_model = None
    _table_wired_structure_split_model = None
    _rapid_undisort_engine = None

    def __init__(self):
        # 存储已加载的模型实例 {model_name: model}
        self._models: Dict[str, Any] = {}
        # 存储模型加载函数 {model_name: loader_func}
        self._loaders: Dict[str, Callable] = {}
        # 存储模型配置 {model_name: config}
        self._configs: Dict[str, Dict] = {}
        # 线程锁，确保懒加载的线程安全
        self._lock = threading.Lock()

    def register_model(
        self, model_name: str, loader: Callable, config: Optional[Dict] = None
    ) -> None:
        """注册模型及其加载函数和配置"""
        self._loaders[model_name] = loader
        self._configs[model_name] = config or {}

    def get_model(self, model_name: str) -> Any:
        """获取模型实例（懒加载）"""
        if model_name not in self._loaders:
            raise ValueError(f"Model {model_name} is not registered!")

        # 检查是否已加载
        if model_name not in self._models:
            with self._lock:  # 加锁防止并发重复加载
                if model_name not in self._models:  # 双重检查锁定
                    try:
                        logger.info(f"Loading model {model_name}...")
                        loader = self._loaders[model_name]
                        config = self._configs[model_name]
                        self._models[model_name] = loader(**config)
                    except Exception as e:
                        logger.error(
                            f"Failed to load model {model_name}: {str(e)}",
                            exc_info=True,
                            stack_info=True,
                        )

        return self._models[model_name]

    def release_model(self, model_name: str) -> None:
        """释放模型资源"""
        if model_name in self._models:
            with self._lock:
                model = self._models.pop(model_name, None)
                if hasattr(model, "close"):
                    model.close()  # 假设模型有释放资源的方法
                logger.info(f"Model {model_name} released.")

    def release_all(self) -> None:
        """释放所有模型资源"""
        for model_name in list(self._models.keys()):
            self.release_model(model_name)

    @staticmethod
    def get_rapid_undistort_engine():
        if ModelManager._rapid_undisort_engine is None:
            ModelManager._rapid_undisort_engine = InferenceEngine()
            logger.info("loading rapid undistort engine")
        return ModelManager._rapid_undisort_engine

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
                ocr=False,
                recovery=False,
                table=False,
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


model_manger = ModelManager()
model_manger.register_model(
    model_name="paddle_locate",
    loader=lambda **kwargs: PPStructure(**kwargs),
    config={
        "show_log": False,
        "table_model_dir": PADDLE_OCR_TABLE_MODEL_DIR,
        "det_model_dir": PADDLE_OCR_DET_MODEL_DIR,
        "rec_model_dir": PADDLE_OCR_REC_MODEL_DIR,
        "use_gpu": True,
        "ocr": False,
        "recovery": False,
        "table": False,
    },
)
model_manger.register_model(
    model_name="yolo_locate",
    loader=lambda **kwargs: YOLO(**kwargs),
    config={
        "model": "./hf_models/yolov8m_table_detection.pt"
    },
)

model_manger.register_model(
    model_name="table_transformer_structure",
    loader=ModelManager.get_table_structure_split_model,
    config={},
)

model_manger.register_model(
    model_name="table_transformer_encoding",
    loader=ModelManager.get_table_structure_feature_extractor_model,
    config={},
)

model_manger.register_model(
    model_name="cyclecenter_net_structure",
    loader=lambda **kwargs: WiredTableStructureModel(**kwargs),
    config={},
)

model_manger.register_model(
    model_name="paddle_rec",
    loader=lambda **kwargs: PaddleOCR(**kwargs),
    config={
        "show_log": False,
        "det_model_dir": PADDLE_OCR_DET_MODEL_DIR,
        "rec_model_dir": PADDLE_OCR_REC_MODEL_DIR,
        "use_gpu": True,
    },
)

model_manger.register_model(
    model_name="bingo_cls",
    loader=lambda **kwargs: Yolov8_cls_PIL(**kwargs),
    config={"model_path": "./hf_models/bingo-cls.onnx"},
)

model_manger.register_model(
    model_name="bingo_det",
    loader=lambda **kwargs: YOLO(**kwargs),
    config={
        "model": "./hf_models/bingo_det.pt"
    },
)


model_manger.register_model(
    model_name="rapid_undistort_engine_preprocess",
    loader=lambda **kwargs: InferenceEngine(),
    config={},
)


def test_modelmanager():
    start = time.time()
    paddle_locate = model_manger.get_model("paddle_locate")
    yolo_locate = model_manger.get_model("yolo_locate")
    table_transformer_encoding = model_manger.get_model("table_transformer_encoding")
    table_transformer_structure = model_manger.get_model("table_transformer_structure")
    cyclecenter_net_structure = model_manger.get_model("cyclecenter_net_structure")
    paddle_rec = model_manger.get_model("paddle_rec")
    bingo_cls = model_manger.get_model("bingo_cls")
    # rapid_undistort_engine_preprocess = model_manger.get_model("rapid_undistort_engine_preprocess")
    # rapid_undistort_engine_preprocess 同时测试 会出现 bad allocation onnxruntime
    model_manger.release_all()
    print("elapsed time", time.time() - start)


if __name__ == "__main__":
    # test_modelmanager()
    pass
