# -*- coding: UTF-8 -*-
"""
@File    ：utils.py
@Author  ：Csy
@Date    ：2024/12/19 15:53 
@Bref    :
@Ref     :
TODO     :
         :
"""

from typing import Union
import os
import subprocess
import platform
import cv2
from PIL import Image, UnidentifiedImageError,ImageDraw,ImageFont
import numpy as np
from pathlib import Path
from Exceptions import *
from io import BytesIO
from adapters.Table import TableEntity
from Settings import *

InputType = Union[str, np.ndarray, bytes, Path]


def open_folder(folder_path):
    if platform.system() == "Windows":
        os.startfile(folder_path)
    elif platform.system() == "Darwin":  # macOS
        subprocess.Popen(["open", folder_path])
    else:  # Linux and others
        subprocess.Popen(["xdg-open", folder_path])


def find_bingo_number(num_list, bing_state: list):
    """
    查找 bingo 对应的数字。

    Args:
        num_list: 包含数字和 "bingo" 的字符串列表。

    Returns:
        如果找到 "bingo"，则返回对应的数字（1-4），否则返回 None。
        如果输入列表格式错误（例如包含非数字或非 "bingo" 的元素），则抛出 ValueError。
    """
    # try:
    #     numbers = [int(num) for num in num_list]
    # except ValueError as e:
    #     # 查找 "bingo" 的索引
    #     try:
    #         bingo_index = num_list.index("bingo")
    #     except ValueError:
    #         raise ValueError("列表格式错误：包含非数字或非 'bingo' 的元素") from e

    #     # 根据索引推断 bingo 对应的数字
    #     if bingo_index < len(num_list):
    #         numbers = [int(num) for num in num_list if num !=
    #                    "bingo"]  # 在except中进行赋值，避免未赋值错误
    #         missing_numbers = set(range(1, 5)) - set(numbers)
    #         if len(missing_numbers) == 1:
    #             return missing_numbers.pop()
    #         else:
    #             raise ValueError("无法确定 bingo 对应的数字，缺失数字数量不为1")
    #     else:
    #         raise ValueError("bingo 位置超出列表长度")

    # return None  # 没有 bingo
    pass


def binarize_images(
    folder_path,
    threshold=127,
    max_value=255,
    method=cv2.THRESH_BINARY,
    adaptive_method=None,
    block_size=11,
    C=2,
):
    """
    对指定文件夹下的图像进行二值化处理。

    Args:
        folder_path: 图像文件夹的路径。
        threshold: 二值化阈值 (仅用于全局阈值方法)。
        max_value: 超过阈值的像素值将被设置为此值。
        method: 全局二值化方法。
        adaptive_method: 自适应二值化方法 (如果使用)。
        block_size: 用于自适应阈值的邻域大小。
        C: 用于自适应阈值的常数，从平均值或加权平均值中减去。
    """
    # ... (之前的代码，检查文件夹和文件类型)
    if not os.path.exists(folder_path):
        print(f"错误：文件夹 '{folder_path}' 不存在。")
        return

    black_percentage_list = []
    filenames = os.listdir(folder_path)
    for filename in filenames:
        if filename.lower().endswith(
            (".png", ".jpg", ".jpeg", ".bmp")
        ):  # 检查文件是否为图像
            image_path = os.path.join(folder_path, filename)
            try:
                img = cv2.imread(image_path)
                if img is None:
                    print(f"警告：无法读取图像 '{filename}'。请检查文件是否损坏。")
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                if adaptive_method is not None:  # 使用自适应阈值
                    binary = cv2.adaptiveThreshold(
                        gray,
                        max_value,
                        adaptive_method,
                        cv2.THRESH_BINARY,
                        block_size,
                        C,
                    )
                elif method == cv2.THRESH_OTSU:  # 使用otsu算法
                    _, binary = cv2.threshold(
                        gray, 0, max_value, method | cv2.THRESH_OTSU
                    )
                else:  # 使用全局阈值
                    _, binary = cv2.threshold(gray, threshold, max_value, method)

                # 滤波
                binary = cv2.medianBlur(binary, 3)

                output_filename = "binary_" + filename  # 输出文件名
                output_path = os.path.join(folder_path, output_filename)
                cv2.imwrite(output_path, binary)
                print(f"{filename} binarized, save to {output_filename}", end=" ")

                total_pixels = binary.size
                white_pixels = cv2.countNonZero(binary)  # 更高效的白色像素计数
                black_pixels = total_pixels - white_pixels

                white_percentage = (white_pixels / total_pixels) * 100
                black_percentage = (black_pixels / total_pixels) * 100
                black_percentage_list.append(black_percentage)
                print("black:white = ", white_percentage, ":", black_percentage)

            except Exception as e:
                print(f"处理图像 '{filename}' 时发生错误：{e}")
    mean = np.array(black_percentage_list).mean()
    std = np.array(black_percentage_list).std()
    print("mean: ", mean, " std: ", std)
    indices_point5std = [
        i
        for i, value in enumerate(black_percentage_list)
        if abs(value - mean) > 0.5 * std
    ]
    indices_1std = [
        i for i, value in enumerate(black_percentage_list) if abs(value - mean) > std
    ]

    print("indices_point5std: ", np.array(filenames)[indices_point5std])
    print("indices_1std: ", np.array(filenames)[indices_1std])


def img_load_by_cv2(img_path: Union[str, Path]):
    # bgr
    verify_image_exist(img_path)
    img = cv2.imdecode(
        np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR
    )  # support chinese
    return img


def img_wirte_by_cv2(img, img_path):
    cv2.imencode(".jpg", img)[1].tofile(img_path)


def img_load_by_Image(img: InputType) -> Image.Image:
    if not isinstance(img, InputType.__args__):
        raise LoadImageError(
            f"The img type {type(img)} does not in {InputType.__args__}"
        )
    if isinstance(img, (str, Path)):
        verify_image_exist(img)
        try:
            img = Image.open(img)
        except UnidentifiedImageError as e:
            raise LoadImageError(f"cannot identify image file {img}") from e
        return img

    if isinstance(img, bytes):
        img = Image.open(BytesIO(img))
        return img

    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
        return img

    raise LoadImageError(f"{type(img)} is not supported!")


def img_convert_to_bgr(img: np.ndarray):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.ndim == 3:
        channel = img.shape[2]
        if channel == 1:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if channel == 2:
            return cvt_two_to_three(img)

        if channel == 4:
            return cvt_four_to_three(img)

        if channel == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        raise LoadImageError(
            f"The channel({channel}) of the img is not in [1, 2, 3, 4]"
        )

    raise LoadImageError(f"The ndim({img.ndim}) of the img is not in [2, 3]")


def cvt_four_to_three(img: np.ndarray) -> np.ndarray:
    """RGBA → BGR"""
    r, g, b, a = cv2.split(img)
    new_img = cv2.merge((b, g, r))

    not_a = cv2.bitwise_not(a)
    not_a = cv2.cvtColor(not_a, cv2.COLOR_GRAY2BGR)

    new_img = cv2.bitwise_and(new_img, new_img, mask=a)
    new_img = cv2.add(new_img, not_a)
    return new_img


def cvt_two_to_three(img: np.ndarray) -> np.ndarray:
    """gray + alpha → BGR"""
    img_gray = img[..., 0]
    img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    img_alpha = img[..., 1]
    not_a = cv2.bitwise_not(img_alpha)
    not_a = cv2.cvtColor(not_a, cv2.COLOR_GRAY2BGR)

    new_img = cv2.bitwise_and(img_bgr, img_bgr, mask=img_alpha)
    new_img = cv2.add(new_img, not_a)
    return new_img


def verify_image_exist(file_path: Union[str, Path]):
    if not Path(file_path).exists():
        raise LoadImageError(f"{file_path} does not exist.")


def intersection(boxA: list, boxB: list):
    """
    计算两个 Box 的交集
    Args:
        boxA: 第一个 Box，(x1, y1, x2, y2)
        boxB: 第二个 Box，(x1, y1, x2, y2)

    Returns:
        如果两个 Box 有交集，返回交集的 Box，否则返回 None
    """

    # 计算交集的左上角和右下角坐标
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # 如果交集的宽度或高度小于等于0，则说明两个 Box 没有交集
    intersection_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    return [xA, yA, xB, yB] if intersection_area > 0 else None


def draw_table(image: Image, table_data: TableEntity,save_dir, **kwargs):
    
    draw_col = kwargs.get("draw_col", False)
    draw_row = kwargs.get("draw_row", False)
    draw_cell = kwargs.get("draw_cell", False)
    cut_cell = kwargs.get("cut_cell", False)

    rows_box_list = table_data.row_bbox_list
    cols_box_list = table_data.col_bbox_list
    cells_box_list = table_data.cell_bbox_list

    if draw_col and len(cols_box_list) > 0:
            col_draw_image = image.copy()
            for col in cols_box_list:
                col_draw = ImageDraw.Draw(col_draw_image)
                col_draw.rectangle(col, outline="red", width=3)
            col_draw_image.save(save_dir + "/" + "cols.jpg")

    if draw_row and len(rows_box_list) > 0:
        row_draw_image = image.copy()
        for row in rows_box_list:
            row_draw = ImageDraw.Draw(row_draw_image)
            row_draw.rectangle(row, outline="red", width=3)
        row_draw_image.save(save_dir + "/" + "rows.jpg")

    if draw_cell and len(cells_box_list) > 0:
        cell_draw_image = image.copy()
        for cell in cells_box_list:
            cell_draw = ImageDraw.Draw(cell_draw_image)
            cell_draw.rectangle(cell, outline="red", width=3)
        cell_draw_image.save(save_dir + "/" + "cells.jpg")

    if draw_cell and (len(cells_box_list) > 0):
        white_background = Image.new(
            "RGB", (image.width, image.height), (255, 255, 255)
        )
        font = ImageFont.truetype(FONT_PATH, size=20)
        sorted_cell_draw = ImageDraw.Draw(white_background)
        for i, cell in enumerate(cells_box_list):
            sorted_cell_draw.rectangle(cell, outline="red", width=3)
            sorted_cell_draw.text(
                xy=(
                    cell[0] + (cell[2] - cell[0]) / 2,
                    cell[1] + (cell[3] - cell[1]) / 2,
                ),
                text=str(i),
                font=font,
                fill=(0, 0, 255),
            )
            sorted_cell_draw.text(
                xy=(
                    cell[0] + (cell[2] - cell[0]) / 2 - 20,
                    cell[1] + (cell[3] - cell[1]) / 2 - 20,
                ),
                text=str(cell[0]) + "_" + str(cell[1]),
                font=font,
                fill=(0, 255, 0),
            )

            white_background.save(save_dir + "/" + "sorted_cells.jpg")
    if cut_cell and (len(cells_box_list) > 0):
        print("enable cutting cells")
        for i, cell in enumerate(cells_box_list):
            x = i // len(cols_box_list)
            y = i % len(cols_box_list)

            cell_image = image.crop(cell)
            cell_image.save(
                save_dir + "/" + f"cell_{x}_{y}" + ".jpg"
            )
        print("cutting cells done")