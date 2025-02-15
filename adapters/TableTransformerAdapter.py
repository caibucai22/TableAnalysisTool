from adapters.ITableAdapter import ITableAdapter
from transformers import TableTransformerForObjectDetection
from adapters.Table import TableEntity
from tools import Utils
from models.ModelManager import ModelManager
import torch
from Settings import *


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

