from service.det.yolo_locate_service import YoloLocateService
from service.det.paddle_locate_service import PaddleLocateService
from service.structure.tabletransformer_service import TableTransformerService
from models.TableProcessModel_v2 import TableProcessModel

from PIL import Image
from tools.Utils import *

from service.custom.ScoreEvaluation_v2 import A4ScoreEvaluation
from Preprocess import A3Split

total_img_path = 'C:/Users/001/Pictures/ocr/v2_test/2-21/back/周瑶.jpg'
# 柳 16 17 : 12
# 周 : 19

def test_back_no_2_table():
    base_dir = "E:/05-OrderProjects/2025/debug_output"
    original_img_shape = Image.open(total_img_path).convert("RGB").size
    no_2_path,no_3_path = A3Split.split(total_img_path,original_img_shape,save_dir=base_dir)

    cur_image_name = os.path.basename(no_2_path).split(".")[0]
    each_img_cache_dir = base_dir + "/" + cur_image_name
    os.makedirs(each_img_cache_dir, exist_ok=True)
    
    img = Image.open(no_2_path).convert("RGB")
    
    yolo_locate_service = YoloLocateService()
    bboxs, roi_imgs = yolo_locate_service.locate_table(no_2_path)
    print(f"found {len(bboxs)} tables")

    table_transformer_service = TableTransformerService()
    score_eval_service = A4ScoreEvaluation()
    for i, bbox in enumerate(bboxs):
        table_img = img.crop(box=bbox)
        table_data = table_transformer_service.recognize_structure(table_img)
        draw_table(
            table_img.copy(),
            table_data,
            save_dir="",
            cut_cell=True,
            draw_cell=True,
            prefix="A3_BACK_NO_2_TABLE",
        )
        n_row, n_col = len(table_data.row_bbox_list), len(table_data.col_bbox_list)
        score_eval_service.load_next(
            table_img,
            table_data.cell_bbox_list,
            image_name=cur_image_name,
            image_type="A3_BACK_NO_2_TABLE",
            save_dir=each_img_cache_dir,
            n_row=n_row,
            n_col=n_col,
        )
        score_eval_service.eval_score()
        score_eval_service.to_xlsx()


def test_back_no_3_table():
    base_dir = "E:/05-OrderProjects/2025/debug_output"
    original_img_shape = Image.open(total_img_path).convert("RGB").size
    no_2_path,no_3_path = A3Split.split(total_img_path,original_img_shape,save_dir=base_dir)

    cur_image_name = os.path.basename(no_3_path).split(".")[0]
    each_img_cache_dir = base_dir + "/" + cur_image_name
    os.makedirs(each_img_cache_dir, exist_ok=True)
    
    img = Image.open(no_3_path).convert("RGB")
    
    yolo_locate_service = YoloLocateService()
    bboxs, roi_imgs = yolo_locate_service.locate_table(no_3_path)
    print(f"found {len(bboxs)} tables")

    table_transformer_service = TableTransformerService()
    score_eval_service = A4ScoreEvaluation()
    for i, bbox in enumerate(bboxs):
        table_img = img.crop(box=bbox)
        table_data = table_transformer_service.recognize_structure(table_img)
        draw_table(
            table_img.copy(),
            table_data,
            save_dir="",
            cut_cell=True,
            draw_cell=True,
            prefix="A3_BACK_NO_3_TABLE",
        )
        n_row, n_col = len(table_data.row_bbox_list), len(table_data.col_bbox_list)
        score_eval_service.load_next(
            table_img,
            table_data.cell_bbox_list,
            image_name=cur_image_name,
            image_type="A3_BACK_NO_3_TABLE",
            save_dir=each_img_cache_dir,
            n_row=n_row,
            n_col=n_col,
        )
        score_eval_service.eval_score()
        score_eval_service.to_xlsx()


if __name__ == "__main__":
    test_back_no_2_table()
    test_back_no_3_table()
