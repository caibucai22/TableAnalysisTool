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
        "logi_points":[]
        }
        """
        cell_bboxs_list = model_output["sorted_polygons"]
        # cell_bboxs_list = sorted(
        #     cell_bboxs_list, key=lambda x: ((x[2] + x[0]) / 2 + (x[3] + x[1]) / 2)
        # )
        adapted_parsed_table = TableEntity([], [], cell_bbox_list=cell_bboxs_list)
        return adapted_parsed_table

