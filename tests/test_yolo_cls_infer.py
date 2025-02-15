from models.YoloClsInfer import Yolov8_cls_PIL

if __name__ == "__main__":
    clser = Yolov8_cls_PIL(
        "E:/Models/ultralytics-8.1.0/train_bingo/bingo-cls/n-base/weights/bingo-cls.onnx"
    )
    result = clser(
        "E:/01-LabProjects-Data/BingoDataset/BingoDataset-cls/train/bingo/cell_6_2.jpg"
    )

    top5_conf = result[0]
    top5_index = result[1]
    for idx in range(len(top5_conf)):
        print("index:{}, conf:{:.2f}".format(top5_index[idx], top5_conf[idx]))
