from adapters.CycleCenterNetAdapter import CycleCenterNetAdapter
from models.WiredTableRecognition import WiredTableStructureModel
from tools.Utils import img_load_by_Image, img_convert_to_bgr, draw_table

import numpy as np
import os


def test_cycle_center_net_adapter():
    adapter = CycleCenterNetAdapter()
    model = WiredTableStructureModel()

    # image_path = "./test_images/table1.jpg"
    image_path = "C:/Users/001/Pictures/ocr/v2/locate_table_4.jpg"
    save_dir, image_name = os.path.split(image_path)
    img = img_load_by_Image(image_path)
    img = img_convert_to_bgr(np.array(img))
    polygons, logits = model(img=img)

    parsed_table = adapter.adapt({"sorted_polygons": polygons})
    print(parsed_table)

    draw_table(img_load_by_Image(image_path), parsed_table, save_dir, draw_cell=True)


if __name__ == "__main__":
    test_cycle_center_net_adapter()
