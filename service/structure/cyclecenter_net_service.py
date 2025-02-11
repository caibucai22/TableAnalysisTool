from typing import Any
from service.interface import ITableStructureService
from models.WiredTableRecognition import WiredTableStructureModel
from adapters.CycleCenterNetAdapter import CycleCenterNetAdapter
from adapters.Table import TableEntity
from models.ModelManager_v2 import model_manger


# 结构识别服务实现
class CycleCenterNetService(ITableStructureService):
    def __init__(self):
        self.model: WiredTableStructureModel = model_manger.get_model(
            "cyclecenter_net_structure"
        )
        self.adapter = CycleCenterNetAdapter()

    def recognize_structure(self, table_image: Any) -> TableEntity:
        polygons, logits = self.model(table_image)
        return self.adapter.adapt({"sorted_polygons": polygons})
