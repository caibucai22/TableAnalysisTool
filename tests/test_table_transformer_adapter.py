from adapters.TableTransformerAdapter import TableTransformerAdapter
from models.ModelManager import ModelManager
from tools.Utils import img_load_by_Image

import torch
from config import load_config

app_config = load_config("config.yaml")


def test_table_transformer_adapter():
    adapter = TableTransformerAdapter()
    table_structure_feature_extractor_model = (
        ModelManager.get_table_structure_feature_extractor_model()
    )
    table_structure_split_model = ModelManager.get_table_structure_split_model()

    image_path = "C:/Users/001/Pictures/ocr/v2/locate_table_2.jpg"
    img = img_load_by_Image(image_path).convert("RGB")
    target_sizes = [img.size[::-1]]
    encoding = table_structure_feature_extractor_model(img, return_tensors="pt")
    if (
        encoding["pixel_values"].device == torch.device("cpu")
        and app_config["app_run"]["use_device"] == "cuda:0"
    ):
        encoding["pixel_values"] = encoding["pixel_values"].cuda()
        encoding["pixel_mask"] = encoding["pixel_mask"].cuda()
    with torch.no_grad():
        outputs = table_structure_split_model(**encoding)
    results = table_structure_feature_extractor_model.post_process_object_detection(
        outputs, threshold=0.85, target_sizes=target_sizes
    )[0]
    parsed_table = adapter.adapt(results)
    print(parsed_table)


if __name__ == "__main__":
    test_table_transformer_adapter()
