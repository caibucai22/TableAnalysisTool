from ITableAdapter import ITableAdapter
from WiredTableRecognition import WiredTableStructureModel
from Table import TableEntity


class CycleCenterNetAdapter(ITableAdapter):

    def __init__(self, model: WiredTableStructureModel) -> None:
        self.model = model

    def adapt(self, model_output: dict, **kwargs) -> TableEntity:
        """
        {
        "sorted_polygons":[],
        "logi_points":[]
        }
        """
        cell_bboxs_list = model_output["sorted_polygons"].tolist()
        adapted_parsed_table = TableEntity([], [], cell_bbox_list=cell_bboxs_list)
        return adapted_parsed_table
