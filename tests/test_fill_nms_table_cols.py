from adapters.TableTransformerAdapter import (
    TableTransformerAdapter,
    process_table_columns,
    visualize_columns,
)
from models.ModelManager import ModelManager
from tools.Utils import img_load_by_Image

import torch
from config import load_config

app_config = load_config("config.yaml")

image_path = "C:/Users/001/Pictures/ocr/v2/locate_table_2.jpg"
image_path = "tests/images/table2.jpg"
img = img_load_by_Image(image_path).convert("RGB")


def test_table_transformer_adapter():
    adapter = TableTransformerAdapter()
    table_structure_feature_extractor_model = (
        ModelManager.get_table_structure_feature_extractor_model()
    )
    table_structure_split_model = ModelManager.get_table_structure_split_model()

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
    return parsed_table


# 示例用法
if __name__ == "__main__":

    parsed_table = test_table_transformer_adapter()
    cols = parsed_table.col_bbox_list
    visualize_columns(cols,img)

    # 测试数据：包含重复列和大间隙
    # test_boxes = [
    #     [10, 50, 40, 200],  # 列1
    #     [45, 55, 75, 190],  # 与列1重复
    #     [150, 60, 180, 210],  # 与前一列有较大间隙
    #     [200, 50, 230, 200],  # 正常列
    #     [235, 60, 265, 190],  # 重复列
    # ]

    processed = process_table_columns(cols)
    print("处理结果：")
    for i, box in enumerate(processed):
        print(
            f"列{i+1}: x1={box[0]:.1f}, y1={box[1]:.1f}, x2={box[2]:.1f}, y2={box[3]:.1f}"
        )
    visualize_columns(processed, img)
