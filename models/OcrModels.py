from models.ModelManager import ModelManager
from paddleocr import PaddleOCR
import tools.Utils as Utils
from typing import Union
from PIL.Image import Image
import numpy as np
import cv2
from tools.Logger import get_logger

logger = get_logger(__file__)


class PaddleOCR_:

    def __init__(self) -> None:
        self.model: PaddleOCR = ModelManager.get_text_rec_model()
        pass

    def infer_cell(self, img: Union[str, Image, np.ndarray], **kwargs):
        """
        OCR with PaddleOCR for table cell
        args：
            img: img for OCR, support ndarray, img_path and list or ndarray
            det: use text detection or not. If False, only rec will be exec. Default is True
            rec: use text recognition or not. If False, only det will be exec. Default is True
            cls: use angle classifier or not. Default is True. If True, the text with rotation of 180 degrees can be recognized. If no text is rotated by 180 degrees, use cls=False to get better performance. Text with rotation of 90 or 270 degrees can be recognized even if cls=False.
            bin: binarize image to black and white. Default is False.
            inv: invert image colors. Default is False.
            alpha_color: set RGB color Tuple for transparent parts replacement. Default is pure white.
        """
        if isinstance(img, str):
            img = Utils.img_load_by_cv2(img)
        if isinstance(img, Image):
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        rec_success = True
        det = kwargs.get("det", False)
        rec = kwargs.get("rec", True)
        cls = kwargs.get("cls", False)
        bin = kwargs.get("bin", False)
        inv = kwargs.get("inv", False)
        ret_ocr = self.model.ocr(img, det=det, rec=rec, cls=cls, bin=bin, inv=inv)
        if ret_ocr[0] is not None:
            ret = ret_ocr[0][0][0]
            cof = ret_ocr[0][0][1]
        else:
            logger.warn("ocr result is none, rec failed")
            rec_success = False
            ret = None
            cof = None
        return rec_success, ret, cof
