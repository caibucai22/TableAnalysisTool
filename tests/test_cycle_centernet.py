from models.WiredTableRecognition import WiredTableStructureModel
from tools.Utils import img_load_by_Image, img_convert_to_bgr

import numpy as np
import time

from config import load_config

app_config = load_config("config.yaml")


def test_cycle_centernet():
    image_path = "./test_images/table1.jpg"
    img = img_load_by_Image(image_path)
    img = img_convert_to_bgr(np.array(img))

    model = WiredTableStructureModel(app_config["app_models"]["cycle_centernet"]["v2"])
    # single test
    now = time.time()
    polygons, logits = model(img=img)  # single test: CUDA 4.2 CPU 1.3
    print(f"elapse {time.time() - now}")
    print(len(polygons))

    # multi test
    image_dir = "C:/Users/001/Pictures/ocr/jpg"
    img_list = [image_dir + "/" + image for image in os.listdir(image_dir)]
    img_inputs = [
        img_convert_to_bgr(np.array(img_load_by_Image(img_input)))
        for img_input in img_list
    ]
    now = time.time()
    for img in img_inputs:
        polygons, logits = model(img=img)  # mean time 0.6s based on 4060 GPU
    print(f"elapse {time.time() - now}")
    print("mean time, ", (time.time() - now) / len(img_list))


if __name__ == "__main__":
    test_cycle_centernet()
