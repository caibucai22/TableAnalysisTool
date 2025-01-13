from ITableAdapter import ITableAdapter
from models.WiredTableRecognition import WiredTableStructureModel
from Table import TableEntity
import Utils


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
        adapted_parsed_table = TableEntity([], [], cell_bbox_list=cell_bboxs_list)
        return adapted_parsed_table


def main():
    adapter = CycleCenterNetAdapter()
    model = WiredTableStructureModel()

    image_path = "../test_images/table1.jpg"
    img = Utils.img_load_by_Image(image_path)
    img = Utils.img_convert_to_bgr(img)
    polygons, logits = model(img=img)

    parsed_table = adapter.adapt({"sorted_polygons": polygons})
    print(parsed_table)


if __name__ == "__main__":
    main()
