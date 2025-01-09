from ITableAdapter import ITableAdapter
from WiredTableRecognition import WiredTableStructureModel


class CycleCenterNetAdapter(ITableAdapter):

    def __init__(self, model: WiredTableStructureModel) -> None:
        self.model = model

    def adapt(self, model_output):
        pass
