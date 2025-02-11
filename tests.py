import os
from tools.Utils import find_bingo_number, binarize_images, check_checkmark
import cv2


def test_check_checkmark():
    # 示例参数
    target_image = "E:/my-github-repos/01-my_repos/base_output/cache/locate_table_2/1.jpg"  # 待检测图像
    template_image = (
        "E:/my-github-repos/01-my_repos/base_output/template.jpg"  # 对号模板
    )

    # 执行检测
    has_checkmark = check_checkmark(
        target_image, template_image, threshold=0.5, visualize=False
    )

    print(f"检测到对号: {has_checkmark}")


def test_find_bingo_number():
    # 示例
    list1 = ["1", "2", "bingo", "4"]
    print(find_bingo_number(list1))  # 输出：3

    list2 = ["4", "3", "2", "bingo"]
    print(find_bingo_number(list2))  # 输出：1

    list3 = ["1", "bingo", "3", "4"]
    print(find_bingo_number(list3))  # 输出：2

    list4 = ["1", "2", "3", "4"]
    print(find_bingo_number(list4))  # 输出：None

    list5 = ["1", "2", "bingo", "bingo"]
    try:
        print(find_bingo_number(list5))
    except ValueError as e:
        print(e)  # 输出：无法确定 bingo 对应的数字，缺失数字数量不为1

    list6 = ["1", "2", "5", "bingo"]
    try:
        print(find_bingo_number(list6))
    except ValueError as e:
        print(e)  # 输出：无法确定 bingo 对应的数字，缺失数字数量不为1

    list7 = ["1", "2", "bingo", "4", "5"]
    try:
        print(find_bingo_number(list7))
    except ValueError as e:
        print(e)  # 输出：bingo 位置超出列表长度

    list8 = ["1", "2", "a", "4"]
    try:
        print(find_bingo_number(list8))
    except ValueError as e:
        print(e)  # 输出：列表格式错误：包含非数字或非 'bingo' 的元素


def test_binarize_images():
    # 使用示例
    folder_path = "E:/05-OrderProjects/2024/12-3-no/cache/table"

    binarize_images(folder_path=folder_path, threshold=1)

    # # 使用自适应均值阈值
    # binarize_images(folder_path, adaptive_method=cv2.ADAPTIVE_THRESH_MEAN_C)

    # # 使用自适应高斯阈值，并调整参数
    # binarize_images(folder_path, adaptive_method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, block_size=15, C=5)

    # # 使用otsu算法
    # binarize_images(folder_path, method=cv2.THRESH_OTSU)

    # # 使用全局阈值，并使用反向阈值
    # binarize_images(folder_path, threshold=80, method=cv2.THRESH_BINARY_INV)


def copy_paste_images(n, image_path, save_dir):
    n = 100
    _, image_name = os.path.split(image_path)
    image_basename, image_ext = os.path.splitext(image_name)
    original_img = cv2.imread(image_path)
    for i in range(n):
        cv2.imwrite(f"{save_dir}/{image_basename}_{i}{image_ext}", original_img)


if __name__ == "__main__":
    # test_find_bingo_number()
    #   test_binarize_images()
    test_check_checkmark()
    # pass
