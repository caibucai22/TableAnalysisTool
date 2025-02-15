import time
import os
from models.TableProcessModel import TableProcessModel


if __name__ == "__main__":
    table_img_path = "C:/Users/001/Pictures/ocr/v2/v2_test_left.jpg"
    image_dir = "./test_images"
    table_img_path_list = [
        image_dir + "/" + imgname
        for imgname in os.listdir(image_dir)
        if os.path.splitext(imgname)[-1] in [".jpg"]
    ]

    t_class_init_start = time.time()
    table_process = TableProcessModel()
    print("model construct elapsed time ", time.time() - t_class_init_start)

    # single_test ~3.5s
    t_single_start = time.time()
    table_process.run(table_img_path)
    print("single test elapsed time ", time.time() - t_single_start)

    # multi_test ~3s
    n = len(table_img_path_list)
    print("found {} images".format(n))
    t_multi_test = time.time()
    for img_path in table_img_path_list:
        table_process.run(img_path)
    print(
        "multi test elapsed time ",
        time.time() - t_multi_test,
        "mean time: ",
        (time.time() - t_multi_test) / n,
    )

    table_process.clear()
