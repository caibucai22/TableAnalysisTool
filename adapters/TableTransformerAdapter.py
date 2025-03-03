from adapters.ITableAdapter import ITableAdapter
from transformers import TableTransformerForObjectDetection
from adapters.Table import TableEntity
from tools import Utils
from Settings import *
import numpy as np
from typing import List, Tuple
from matplotlib import pyplot as plt
from tools.Utils import filter_by_mean_std
from tools.Logger import get_logger

logger = get_logger(__file__)


def process_table_columns(
    col_boxes: List[List[float]],
    gap_threshold_ratio: float = 1.5,
    nms_iou_threshold: float = 0.3,
) -> List[List[float]]:
    """
    处理表格列检测结果，包含补列和去重功能

    参数：
    col_boxes -- 模型输出的列框列表，每个框格式为[x1,y1,x2,y2]
    gap_threshold_ratio -- 间隙补列阈值比例（相对于平均列宽）
    nms_iou_threshold -- NMS去重的IOU阈值

    返回：
    处理后的列框列表
    """
    print(f"before processing, total {len(col_boxes)} cols")
    # 预处理：确保框按从左到右排序
    sorted_boxes = sorted(col_boxes, key=lambda x: x[0])

    # 第一阶段：补列处理
    filled_boxes = fill_missing_columns(sorted_boxes, gap_threshold_ratio)
    print(f"filling {len(filled_boxes) - len(col_boxes)} cols")

    # 第二阶段：去重处理
    final_boxes = apply_adaptive_nms(filled_boxes, nms_iou_threshold)
    print(f"filtering {len(filled_boxes) - len(final_boxes)} cols")

    return sorted(final_boxes, key=lambda x: x[0])


def fill_missing_columns(
    boxes: List[List[float]], threshold_ratio: float
) -> List[List[float]]:
    """自动补充分隔过大的列间隙"""
    if len(boxes) < 2:
        return boxes
    visualize_columns(boxes)
    filled = []
    prev_box = boxes[0]
    filled.append(prev_box)

    # 计算平均列宽和行高
    avg_width = np.mean([b[2] - b[0] for b in boxes])
    avg_height = np.mean([b[3] - b[1] for b in boxes])

    for current_box in boxes[1:]:
        gap = current_box[0] - prev_box[2]
        prev_width = prev_box[2] - prev_box[0]

        # 动态阈值：考虑前后列宽度的平均值
        dynamic_threshold = (
            (prev_width + (current_box[2] - current_box[0])) / 2 * threshold_ratio
        )

        if gap > dynamic_threshold:
            # 计算缺失列的数量（考虑可能连续缺失多列）
            missing_count = int(round(gap / (avg_width * 1.2)))  # 1.2为宽松系数
            if missing_count < 1:
                missing_count = 1

            # 生成缺失列的位置（线性插值）
            start_x = prev_box[2]
            end_x = current_box[0]
            # step = (end_x - start_x) / (missing_count + 1)
            step = (end_x - start_x) / missing_count

            for i in range(1, missing_count + 1):
                new_x1 = start_x + step * (
                    i - 1
                )  # * (i-1) 从当前 star_x 开始 而不要 再加一个开始
                new_x2 = new_x1 + avg_width
                # 保持y坐标与上下文一致
                new_y1 = min(prev_box[1], current_box[1])
                new_y2 = max(prev_box[3], current_box[3])
                filled.append(
                    [
                        new_x1,
                        new_y1 - avg_height * 0.1,  # 扩展10%高度
                        new_x2,
                        new_y2 + avg_height * 0.1,
                    ]
                )

        filled.append(current_box)
        prev_box = current_box

    return filled


def apply_adaptive_nms(
    boxes: List[List[float]], iou_threshold: float
) -> List[List[float]]:
    """自适应NMS去重，考虑列的特殊性"""
    if len(boxes) < 2:
        return boxes

    # 转换为numpy数组
    boxes_array = np.array(boxes)
    x_centers = (boxes_array[:, 0] + boxes_array[:, 2]) / 2
    scores = np.ones(len(boxes))  # 伪分数，假设所有框置信度相同

    # 按中心点位置排序
    sorted_indices = np.argsort(x_centers)
    boxes_sorted = boxes_array[sorted_indices]

    keep = []
    while boxes_sorted.size > 0:
        # 取第一个框
        current = boxes_sorted[0]
        keep.append(current.tolist())

        if boxes_sorted.size == 1:
            break

        # 计算与后续框的IOU（仅考虑水平方向）
        current_x1 = current[0]
        current_x2 = current[2]

        next_x1 = boxes_sorted[1:, 0]
        next_x2 = boxes_sorted[1:, 2]

        # 水平方向的IOU计算
        xx1 = np.maximum(current_x1, next_x1)
        xx2 = np.minimum(current_x2, next_x2)
        inter = np.maximum(0.0, xx2 - xx1)

        current_width = current_x2 - current_x1
        next_widths = next_x2 - next_x2
        union = current_width + (next_x2 - next_x1) - inter

        iou = inter / (union + 1e-8)

        # 保留IOU低于阈值的框（删除高重叠的框）
        keep_indices = np.where(iou < iou_threshold)[0] + 1  # +1因为跳过第一个
        boxes_sorted = boxes_sorted[keep_indices]

    return keep


def visualize_columns(boxes, image=None, title="Columns Visualization", show=False):
    plt.figure(figsize=(12, 6))
    if image is not None:
        plt.imshow(image)

    colors = ["red", "lime", "blue", "yellow", "cyan"]
    for i, box in enumerate(boxes):
        color = colors[i % len(colors)]
        plt.gca().add_patch(
            plt.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                fill=True,
                edgecolor=color,
                linewidth=2,
                label=f"Col {i+1}",
            )
        )
        plt.text(
            box[0], box[1] - 5, f"{i+1}", color=color, fontweight="bold", fontsize=8
        )

    plt.legend(loc="upper right")
    plt.title(title)
    if show:
        plt.show()


class TableTransformerAdapter(ITableAdapter):

    def __init__(self, model: TableTransformerForObjectDetection = None) -> None:
        self.model = model

    def adapt(self, model_output: dict, **kwargs) -> TableEntity:
        """
        {0: 'table', 1: 'table column', 2: 'table row', 3: 'table column header',
        4: 'table projected row header', 5: 'table spanning cell'}
        """
        cols_box_list = [
            model_output["boxes"][i].tolist()
            for i in range(len(model_output["boxes"]))
            if model_output["labels"][i].item() == 1
        ]
        # 处理少列 和 多列情况 发生与 gt 不匹配时 进行处理
        # 2025/2/28 优化
        if "gt_table_config" in kwargs.keys():
            gt_table_config = kwargs.get("gt_table_config")
            eval_type = gt_table_config["eval_type"]
            gt_cols = gt_table_config["cols"]
            # gt_rows = gt_table_config["rows"]
            logger.info(f"current table cols: {len(cols_box_list)}, gt_cols: {gt_cols}")
            # 2025/3/2 优化 更加细致的处理
            if len(cols_box_list) < gt_cols:
                cols_box_list = fill_missing_columns(
                    cols_box_list, threshold_ratio=0.8
                )  # 默认1列
                visualize_columns(cols_box_list)
            elif len(cols_box_list) > gt_cols:
                for n in [0.5, 1, 1.5, 2]:
                    logger.info(f"filtering by mean and std, current {n}_std")
                    cols_box_list_ = filter_by_mean_std(cols_box_list, n_std=n)
                    if len(cols_box_list_) == gt_cols:
                        cols_box_list = filter_by_mean_std(cols_box_list, n_std=1.5)
                        break
            logger.info(
                f"after processing ,current table cols: {len(cols_box_list)}, gt_cols: {gt_cols}"
            )

        # TODO cols_nms
        rows_box_list = [
            model_output["boxes"][i].tolist()
            for i in range(len(model_output["boxes"]))
            if model_output["labels"][i].item() == 2
        ]

        rows_box_list = sorted(
            rows_box_list, key=lambda x: ((x[2] + x[0]) / 2 + (x[3] + x[1]) / 2)
        )
        cols_box_list = sorted(
            cols_box_list, key=lambda x: ((x[2] + x[0]) / 2 + (x[3] + x[1]) / 2)
        )
        cells_box_list = [
            Utils.intersection(row, col)
            for row in rows_box_list
            for col in cols_box_list
        ]

        parsed_table = TableEntity(
            row_bbox_list=rows_box_list,
            col_bbox_list=cols_box_list,
            cell_bbox_list=cells_box_list,
        )
        return parsed_table
