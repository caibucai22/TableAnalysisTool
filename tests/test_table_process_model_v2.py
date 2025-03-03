import time
import os
from models.TableProcessModel_v2 import TableProcessModel
from service.registry import registry


def test_a3_direct_rec_by_cell():
    pass


if __name__ == "__main__":
    table_img_path = "C:/Users/001/Pictures/ocr/v2/v2_test_left.jpg"
    table_img_path = "C:/Users/001/Pictures/ocr/v2/locate_table_3.jpg"
    table_img_path = "C:/Users/001/Pictures/ocr/v2/a3_single_test2.jpg"
    table_img_path = "C:/Users/001/Pictures/ocr/v2/微信图片_20250117213130.jpg"
    table_img_path = "C:/Users/001/Pictures/ocr/v2_test/2-21/柳悦.jpg"
    table_img_path = "C:/Users/001/Pictures/ocr/v2_test/2-24/周.jpg"

    image_dir = "./test_images"
    table_img_path_list = [
        image_dir + "/" + imgname
        for imgname in os.listdir(image_dir)
        if os.path.splitext(imgname)[-1] in [".jpg"]
    ]

    t_class_init_start = time.time()
    table_process = TableProcessModel(service_registry=registry)
    print("model construct elapsed time ", time.time() - t_class_init_start)

    # single_test ~3.5s
    t_single_start = time.time()
    table_process.run(table_img_path, action="a3_eval")
    print("single test elapsed time ", time.time() - t_single_start)
    table_process.a4table_score_eval_service.score_history_to_xlsx()
    # print(table_process.a4table_score_eval_service.action_xlsx_history.keys())
    # for (
    #     key,
    #     type_history,
    # ) in table_process.a4table_score_eval_service.action_xlsx_history.items():
    #     print(key, len(type_history))
    table_process.a4table_score_eval_service.action_score_hisory_to_xlsx()
    # table_process.export_peroson_info()

    # multi_test ~3s
    # n = len(table_img_path_list)
    # print("found {} images".format(n))
    # t_multi_test = time.time() 
    # for img_path in table_img_path_list:
    #     table_process.run(img_path, action="a3_eval_back")
    # print(
    #     "multi test elapsed time ",
    #     time.time() - t_multi_test,
    #     "mean time: ",
    #     (time.time() - t_multi_test) / n,
    # )

    pass
