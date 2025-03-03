from service.det.yolo_locate_service import YoloLocateService
from service.det.paddle_locate_service import PaddleLocateService
from service.structure.tabletransformer_service import TableTransformerService
from service.structure.cyclecenter_net_service import CycleCenterNetService
from models.TableProcessModel_v2 import TableProcessModel

from PIL import Image
from tools.Utils import *

from service.custom.ScoreEvaluation_v2 import A4ScoreEvaluation
from Preprocess import A3Split

from config import load_config

app_config = load_config("config.yaml")
table_config = load_config("service/table_config.yaml")

total_img_path = "C:/Users/001/Pictures/ocr/v2_test/2-21/周瑶.jpg"
# 周 表2 未检测出 bingo 默认index 是 0 导致结果为 1
total_img_path = "C:/Users/001/Pictures/ocr/v2_test/2-21/柳悦.jpg"
# 柳 表2 1->出错 0 2->未检测出 bingo 默认index 是 0 导致结果为 1

total_img_path = "C:/Users/001/Pictures/ocr/v2_test/2-24/柳.jpg"
# 柳 表1 1->出错 A误识别为C 
# 表2 1 2->未检测出 bingo 结果却为 4 

total_img_path = "C:/Users/001/Pictures/ocr/v2_test/2-24/周.jpg"
# 柳 表1 1->出错 A误识别为C 
# 表2 1 2->未检测出 bingo 结果却为 4 

base_dir = "E:/05-OrderProjects/2025/debug_output/two_structure_model"
original_img_shape = Image.open(total_img_path).convert("RGB").size
left_path, rght_path = A3Split.split(
    total_img_path, original_img_shape, save_dir=base_dir
)

cur_image_name = os.path.basename(rght_path).split(".")[0]
each_img_cache_dir = base_dir + "/" + cur_image_name
os.makedirs(each_img_cache_dir, exist_ok=True)

img = Image.open(rght_path).convert("RGB")
table_transformer_service = TableTransformerService()
cyclecenter_structure_servie = CycleCenterNetService()
score_eval_service = A4ScoreEvaluation()


def test_rght_no_1234_table():
    yolo_locate_service = YoloLocateService()
    bboxs, roi_imgs = yolo_locate_service.locate_table(rght_path)
    print(f"found {len(bboxs)} tables")
    locate_table_type = [
        "A3_RIGHT_NO_1_TABLE",
        "A3_RIGHT_NO_2_TABLE",
        "A3_RIGHT_NO_3_TABLE",
        "A3_RIGHT_NO_4_TABLE",
    ]

    for i, bbox in enumerate(bboxs):
        table_img = img.crop(adjst_box(bbox, ratio=0.1))
        if i == 3:
            table_data = cyclecenter_structure_servie.recognize_structure(
                img_convert_to_bgr(np.array(table_img))
            )
        else:
            table_data = table_transformer_service.recognize_structure(table_img)
        # if i < 3:
        draw_table(
            table_img.copy(),
            table_data,
            save_dir=each_img_cache_dir,
            cut_cell=True,
            draw_cell=True,
            prefix=locate_table_type[i],
        )
        n_row, n_col = len(table_data.row_bbox_list), len(table_data.col_bbox_list)
        if i < 3:
            score_eval_service.load_next(
                table_img,
                table_data.cell_bbox_list,
                image_name=cur_image_name,
                image_type=locate_table_type[i],
                save_dir=each_img_cache_dir,
                n_row=n_row,
                n_col=n_col,
            )
            score_eval_service.eval_score()
            score_eval_service.to_xlsx()
        else:
            gt_cell_idxs = table_config["A3"]["right_no_4_table"]["eval"][
                "gt_cell_idxs"
            ]  # 经测试方能确定的超参数
            gt_names = table_config["A3"]["right_no_4_table"]["eval"]["gt_names"]
            rec_dict = {}

            for i, (name, idx) in enumerate(zip(gt_names, gt_cell_idxs)):
                eval_cell = table_img.crop(
                    adjst_box(table_data.cell_bbox_list[idx], enlarge=False)
                )
                cell_rec_ret = score_eval_service.rec_single_cell(eval_cell)
                if cell_rec_ret["state"]:
                    rec_dict[name] = cell_rec_ret["txt"]
                else:
                    rec_dict[name] = "no rec"
            for key, value in rec_dict.items():
                print(f"{key}: {value}")


def test_rght_no_1234_table2():
    yolo_locate_service = YoloLocateService()
    bboxs, roi_imgs = yolo_locate_service.locate_table(rght_path)
    print(f"found {len(bboxs)} tables")
    locate_table_type = [
        "A3_RIGHT_NO_1_TABLE",
        "A3_RIGHT_NO_2_TABLE",
        "A3_RIGHT_NO_3_TABLE",
        "A3_RIGHT_NO_4_TABLE",
    ]

    for i, bbox in enumerate(bboxs):
        i = 2
        bbox = bboxs[i]
        table_img = img.crop(adjst_box(bbox, ratio=0.1))
        
        cycle_table_data = cyclecenter_structure_servie.recognize_structure(
            img_convert_to_bgr(np.array(table_img)))
        
        trans_table_data = table_transformer_service.recognize_structure(table_img)
        # if i < 3:
        draw_table(
            table_img.copy(),
            cycle_table_data,
            save_dir=each_img_cache_dir,
            cut_cell=False,
            draw_cell=True,
            prefix="cycle_"+locate_table_type[i],
        )
        draw_table(
            table_img.copy(),
            trans_table_data,
            save_dir=each_img_cache_dir,
            cut_cell=False,
            draw_cell=True,
            prefix="trans_"+locate_table_type[i],
        )
        # n_row, n_col = len(table_data.row_bbox_list), len(table_data.col_bbox_list)
        # if i < 3:
        #     score_eval_service.load_next(
        #         table_img,
        #         table_data.cell_bbox_list,
        #         image_name=cur_image_name,
        #         image_type=locate_table_type[i],
        #         save_dir=each_img_cache_dir,
        #         n_row=n_row,
        #         n_col=n_col,
        #     )
        #     score_eval_service.eval_score()
        #     score_eval_service.to_xlsx()
        # else:
        #     gt_cell_idxs = table_config["A3"]["right_no_4_table"]["eval"][
        #         "gt_cell_idxs"
        #     ]  # 经测试方能确定的超参数
        #     gt_names = table_config["A3"]["right_no_4_table"]["eval"]["gt_names"]
        #     rec_dict = {}

        #     for i, (name, idx) in enumerate(zip(gt_names, gt_cell_idxs)):
        #         eval_cell = table_img.crop(
        #             adjst_box(table_data.cell_bbox_list[idx], enlarge=False)
        #         )
        #         cell_rec_ret = score_eval_service.rec_single_cell(eval_cell)
        #         if cell_rec_ret["state"]:
        #             rec_dict[name] = cell_rec_ret["txt"]
        #         else:
        #             rec_dict[name] = "no rec"
        #     for key, value in rec_dict.items():
        #         print(f"{key}: {value}")

        break


if __name__ == "__main__":
    test_rght_no_1234_table2()
