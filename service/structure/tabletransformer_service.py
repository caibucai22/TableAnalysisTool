from typing import Any
from service.interface import ITableStructureService
from models.TableTransformer import TableTransformer
from adapters.TableTransformerAdapter import TableTransformerAdapter
from adapters.Table import TableEntity


# 结构识别服务实现
class TableTransformerService(ITableStructureService):
    def __init__(self):
        self.model = TableTransformer()
        self.adapter = TableTransformerAdapter()

    def recognize_structure(self, table_image: Any,**kwargs) -> TableEntity:
        result_dict = self.model.infer(table_image)
        return self.adapter.adapt(result_dict,**kwargs)
