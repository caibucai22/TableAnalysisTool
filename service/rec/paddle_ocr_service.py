from service.interface import IOCRService
from models.ModelManager_v2 import model_manger
from models.OcrModels import PaddleOCR_
import tools.Utils as Utils


# OCR服务实现（示例）
class PaddleOCRService(IOCRService):
    def __init__(self):
        # 初始化OCR模型（需补充实际实现）
        self.model: PaddleOCR_ = PaddleOCR_()

    def recognize_text(self, cell_images: list) -> list:
        rec_list = []
        for img in cell_images:
            res = {}
            rec_success, ret, cof = self.model.infer_cell(img)
            if rec_success:
                res["state"] = True
                res["txt"] = ret
                res["conf"] = cof
            else:
                res["state"] = False
            rec_list.append(res)
        return rec_list
