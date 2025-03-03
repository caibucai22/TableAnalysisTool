from service.det.yolo_locate_service import YoloLocateService
from service.det.paddle_locate_service import PaddleLocateService
from service.structure.tabletransformer_service import TableTransformerService
from service.structure.cyclecenter_net_service import CycleCenterNetService
from models.TableProcessModel_v2 import TableProcessModel

from PIL import Image
from tools.Utils import *

from service.custom.ScoreEvaluation_v2 import A4ScoreEvaluation
from Preprocess import A3Split


total_img_path = "C:/Users/001/Pictures/ocr/v2_test/2-24/周.jpg"

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


def test_two_model_parsed():
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
        if i == 3:
            continue
        i = 2
        bbox = bboxs[i]
        table_img1 = img.crop(adjst_box(bbox, ratio=0.0))
        table_img2 = img.crop(adjst_box(bbox, ratio=0.0))

        table_data1 = cyclecenter_structure_servie.recognize_structure(
            img_convert_to_bgr(np.array(table_img1))
        )

        table_data2 = table_transformer_service.recognize_structure(table_img2)

        draw_table(
            table_img1.copy(),
            table_data1,
            save_dir=each_img_cache_dir,
            draw_cell=True,
            prefix="cyclecenter_" + locate_table_type[i],
        )

        draw_table(
            table_img2.copy(),
            table_data2,
            save_dir=each_img_cache_dir,
            draw_cell=True,
            prefix="table_transformer_" + locate_table_type[i],
        )
        # cyclecenternet 更精准 table-transformer 多余框更少

        #
        # bbox 超过 > 图片 宽 一半 过滤
        # 5行
        # bbox < 图片 长 1/7 过滤

        # cyclecenternet nms 自行过滤一次

        """
        
        """
        cycle_cells_list = table_data1.cell_bbox_list
        cycle_confs_list = table_data1.cell_conf_list

        trans_cells_list = table_data2.cell_bbox_list

        print(f"cycle_cells : {len(cycle_cells_list)}")
        print(f"trans_cells : {len(trans_cells_list)}")

        cycle_cells_filtered1, cycle_confs_filtered1 = filter_by_w_h(
            cycle_cells_list,
            table_img1.size,
            (1 / 7, 2 / 3),
            (1 / 7, 2 / 3)
        )
        cycle_cells_final = nms(
            cycle_cells_filtered1, iou_threshold=0.25
        )

        trans_cells_filtered1, _ = filter_by_w_h(
            trans_cells_list, table_img2.size, (1 / 7, 2 / 3), (1 / 7, 2 / 3)
        )

        print(f"cycle_cells_final : {len(cycle_cells_final)}")
        print(f"trans_cells_final : {len(trans_cells_filtered1)}")

        draw_table(
            table_img1.copy(),
            TableEntity([], [], cycle_cells_final),
            save_dir=each_img_cache_dir,
            draw_cell=True,
            prefix="cyclecenter_filtered" + locate_table_type[i],
        )

        draw_table(
            table_img1.copy(),
            TableEntity([], [], trans_cells_filtered1),
            save_dir=each_img_cache_dir,
            draw_cell=True,
            prefix="trans_filtered" + locate_table_type[i],
        )

        break


def filter_by_w_h(
    cells, img_shape: tuple, w_range: tuple, h_range: tuple, confs: list = None
):
    w, h = img_shape  # Image size
    min_w_ratio, max_w_ratio = w_range
    min_h_ratio, max_h_ratio = h_range
    min_w, max_w = min_w_ratio * w, max_w_ratio * w
    min_h, max_h = min_h_ratio * h, max_h_ratio * h
    print(f"w_range: {min_w} ~ {max_w}")
    print(f"h_range: {min_h} ~ {max_h}")
    filtered_cells = []
    filtered_confs = []
    for i, cell in enumerate(cells):
        cell_w = cell[2] - cell[0]
        cell_h = cell[3] - cell[1]
        if min_w <= cell_w <= max_w and min_h <= cell_h <= max_h:
            print(cell_w,f" in w_range: {min_w} ~ {max_w}")
            print(cell_h,f" in h_range: {min_h} ~ {max_h}")
            filtered_cells.append(cell)
            if confs is not None:
                filtered_confs.append(confs[i])
    return filtered_cells, filtered_confs


def nms(cells, confs=None, iou_threshold=0.5):
    if not cells:
        return []
    if confs != None:
        cells_np = np.array(cells)
        confs_np = np.array(confs)
        order = confs_np.argsort()[::-1]
        cells = [cells_np[i].tolist() for i in order]

    keep = []
    while cells:
        # Take the cell with the highest confidence
        best_cell = cells.pop(0)
        keep.append(best_cell)

        # Compute IoU of the remaining cells with the best cell
        cells = [cell for cell in cells if iou(best_cell, cell) < iou_threshold]
        print("[nms] cur loop, save ",len(cells)," cell")
    return keep


def iou(box1, box2):
    """Compute the Intersection over Union (IoU) of two bounding boxes."""
    if len(box1) == 5:
        x1, y1, w1, h1, _ = box1
        x2, y2, w2, h2, _ = box2
    elif len(box1) == 4:
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
    else:
        raise Exception("box 格式不支持")

    # Intersection rectangle
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)

    # Intersection area
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Union area
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    # IoU
    iou_score = inter_area / union_area if union_area != 0 else 0
    return iou_score


if __name__ == "__main__":
    test_two_model_parsed()
