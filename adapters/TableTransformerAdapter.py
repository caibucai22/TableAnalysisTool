from ITableAdapter import ITableAdapter
from transformers import TableTransformerForObjectDetection


class TableTransformerAdapter(ITableAdapter):

    def __init__(self, model: TableTransformerForObjectDetection) -> None:
        self.model = model

    def adapt(self, model_output):
        pass
