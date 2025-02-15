from models.TableLocateModels import YoloLocate, PaddleLocate
import os
from tools.Utils import draw_locate


def test_yoloLocate():
    model = YoloLocate()
    # img_path = "C:/Users/001/Pictures/ocr/v2/right.jpg"
    img_path = "C:/Users/001/Pictures/ocr/v2/locate_table_1.jpg"
    save_dir, image_name = os.path.split(img_path)
    image_basename, _ = os.path.splitext(image_name)
    bboxs, _ = model.infer(img=img_path)
    draw_locate(img_path, bboxs=bboxs, save_dir=save_dir, cut=True)


def test_paddleLocate():
    model = PaddleLocate()
    img_path = "C:/Users/001/Pictures/ocr/v2/right.jpg"
    save_dir, image_name = os.path.split(img_path)
    bboxs, roi_imgs = model.infer(img_path)
    print(f"total {len(bboxs)} tables")
    draw_locate(img_path, bboxs=bboxs, save_dir=save_dir, cut=True, adjust_ratio=0.0)


if __name__ == "__main__":
    test_paddleLocate()
    test_yoloLocate()
