from adapters.ITableAdapter import ITableAdapter
from models.WiredTableRecognition import WiredTableStructureModel
from adapters.Table import TableEntity
from tools.Utils import draw_table, img_load_by_Image, img_convert_to_bgr
import numpy as np
import os


class CycleCenterNetAdapter(ITableAdapter):

    def __init__(self, model: WiredTableStructureModel = None) -> None:
        self.model = model

    def adapt(self, model_output: dict, **kwargs) -> TableEntity:
        """
        {
        "sorted_polygons":[],
        "logits":[]
        }
        """
        table_type: str = kwargs.get("table_type", None)
        cell_bboxs_list = model_output["sorted_polygons"]
        conf_list = model_output["logits"]
        # cell_bboxs_list = sorted(
        #     cell_bboxs_list, key=lambda x: ((x[2] + x[0]) / 2 + (x[3] + x[1]) / 2)
        # )
        cell_bboxs_list_with_conf = [
            [a[0], a[1], a[2], a[3], b] for a, b in zip(cell_bboxs_list, conf_list)
        ]
        # 2025/2/28 优化：过滤 cell_bboxs_list  通过 长宽 当长 宽接近图像比例时 过滤 掉

        if table_type.startswith("A3_RIGHT"):
            if "img" in kwargs.keys():
                img_h, img_w = kwargs.get("img").shape[:2]  # cv2 img
                widths = [b[2] - b[0] for b in cell_bboxs_list]
                heights = [b[3] - b[1] for b in cell_bboxs_list]
                filtered_cell_bboxs = []
                for i, cell in enumerate(cell_bboxs_list):
                    if widths[i] > 0.9 * img_w:
                        continue

                    if heights[i] > 0.9 * img_h or heights[i] < 0.05 * img_h:
                        continue

                    filtered_cell_bboxs.append(cell)
                cell_bboxs_list = filtered_cell_bboxs
        adapted_parsed_table = TableEntity(
            [], [], cell_bbox_list=cell_bboxs_list, cell_conf_list=conf_list
        )
        return adapted_parsed_table

