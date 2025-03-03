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

from typing import Union, List
import os
import subprocess
import platform
import cv2
from PIL import Image, UnidentifiedImageError, ImageDraw, ImageFont
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from tools.Exceptions import *
from io import BytesIO
from adapters.Table import TableEntity

# from Settings import *

InputType = Union[str, np.ndarray, bytes, Path]

import logging
from tools.Logger import get_logger

logger = get_logger(__file__, log_level=logging.INFO)


import cv2
import numpy as np

from config import load_config

app_config = load_config("config.yaml")


def check_checkmark(
    target_img_path: str,
    template_img_path: str,
    threshold: float = 0.8,
    visualize: bool = False,
) -> bool:
    """
    使用模板匹配检测目标图像中是否存在对号（勾选符号）

    参数:
        target_img_path (str): 目标图像路径
        template_img_path (str): 对号模板图像路径
        threshold (float): 匹配阈值（0~1），默认0.8
        visualize (bool): 是否可视化匹配结果，默认False

    返回:
        bool: 是否存在对号
    """
    # 读取图像并转换为灰度图
    target_img = cv2.imread(target_img_path)
    template_img = cv2.imread(template_img_path)

    if target_img is None or template_img is None:
        raise ValueError("图像读取失败，请检查路径是否正确")

    target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)

    # 获取模板尺寸
    h, w = template_gray.shape

    # 执行模板匹配（使用归一化相关系数匹配法）
    result = cv2.matchTemplate(target_gray, template_gray, cv2.TM_CCOEFF_NORMED)

    # 获取最大匹配值和位置
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # 可视化匹配结果（调试使用）
    if visualize:
        # 绘制匹配区域矩形框
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(target_img, top_left, bottom_right, (0, 255, 0), 2)

        # 显示匹配结果
        cv2.imshow("Detection Result", target_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 判断是否超过阈值
    return max_val >= threshold


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


def images_convert(
    input_images: List[Union[str, Image.Image, np.ndarray]], to_type: str
) -> List[np.ndarray]:
    """
    Convert a list of images into the specified type.

    Args:
        input_images (List[Union[str, Image.Image, np.ndarray]]): A list of images, which could be file paths, PIL Image, or np.ndarray.
        to_type (str): The desired output type, either 'image' for PIL Image or 'cv2' for np.ndarray.

    Returns:
        List[np.ndarray]: A list of images in the desired format (np.ndarray or PIL Image).

    Raises:
        ValueError: If the input list contains invalid types or the conversion is not possible.
    """
    converted_images = []

    for img in input_images:
        if isinstance(img, str):  # Path (supporting Chinese paths)
            # Ensure the path is valid, and open the image
            img_path = os.path.abspath(img)
            if os.path.exists(img_path):
                # Open the image based on path, cv2 for 'cv2' type, PIL for 'image' type
                if to_type == "Image":
                    pil_img = Image.open(img_path).convert("RGB")
                    converted_images.append(pil_img)
                elif to_type == "cv2":
                    cv2_img = cv2.imdecode(
                        np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR
                    )
                    if cv2_img is None:
                        raise ValueError(f"Failed to load image from path: {img_path}")
                    converted_images.append(cv2_img)
                else:
                    raise ValueError("Invalid target type. Must be 'image' or 'cv2'.")
            else:
                raise ValueError(f"File does not exist: {img_path}")
        elif isinstance(img, Image.Image):  # PIL Image
            if to_type == "Image":
                converted_images.append(img)
            elif to_type == "cv2":
                # Convert PIL Image to cv2 (numpy.ndarray)
                cv2_img = np.array(img)
                cv2_img = cv2.cvtColor(
                    cv2_img, cv2.COLOR_RGB2BGR
                )  # Convert RGB to BGR if needed
                converted_images.append(cv2_img)
            else:
                raise ValueError("Invalid target type. Must be 'image' or 'cv2'.")
        elif isinstance(img, np.ndarray):  # cv2 image
            if to_type == "cv2":
                converted_images.append(img)
            elif to_type == "Image":
                pil_img = Image.fromarray(
                    cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                )  # Convert BGR to RGB for PIL
                converted_images.append(pil_img)
            else:
                raise ValueError("Invalid target type. Must be 'image' or 'cv2'.")
        else:
            raise ValueError(
                "Invalid input type. Must be path (str), PIL Image, or np.ndarray."
            )

    return converted_images


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


def draw_table(image: Image, table_data: TableEntity, save_dir, **kwargs):

    draw_col = kwargs.get("draw_col", False)
    draw_row = kwargs.get("draw_row", False)
    draw_cell = kwargs.get("draw_cell", False)
    cut_cell = kwargs.get("cut_cell", False)
    prefix = kwargs.get("prefix", "")

    rows_box_list = table_data.row_bbox_list
    cols_box_list = table_data.col_bbox_list
    cells_box_list = table_data.cell_bbox_list

    if draw_col and len(cols_box_list) > 0:
        col_draw_image = image.copy()
        for col in cols_box_list:
            col_draw = ImageDraw.Draw(col_draw_image)
            col_draw.rectangle(col, outline="red", width=3)
        col_draw_image.save(save_dir + "/" + f"{prefix}_cols.jpg")

    if draw_row and len(rows_box_list) > 0:
        row_draw_image = image.copy()
        for row in rows_box_list:
            row_draw = ImageDraw.Draw(row_draw_image)
            row_draw.rectangle(row, outline="red", width=3)
        row_draw_image.save(save_dir + "/" + f"{prefix}_rows.jpg")

    if draw_cell and len(cells_box_list) > 0:
        cell_draw_image = image.copy()
        for cell in cells_box_list:
            cell_draw = ImageDraw.Draw(cell_draw_image)
            cell_draw.rectangle(cell, outline="red", width=3)
        cell_draw_image.save(save_dir + "/" + f"{prefix}_cells.jpg")

    if draw_cell and (len(cells_box_list) > 0):
        white_background = Image.new(
            "RGB", (image.width, image.height), (255, 255, 255)
        )
        font = ImageFont.truetype(
            app_config["app_dir"]["resources_dir"]["font"], size=20
        )
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

            white_background.save(save_dir + "/" + f"{prefix}_sorted_cells.jpg")
    if cut_cell and (len(cells_box_list) > 0):
        logger.info("enable cutting cells")
        for i, cell in enumerate(cells_box_list):
            try:
                cell_image = image.crop(cell)
                x = i // len(cols_box_list)
                y = i % len(cols_box_list)
                cell_image.save(save_dir + "/" + f"cell_{x}_{y}" + ".jpg")
            except Exception as e:
                logger.error(f"{e}")
                logger.info("don't undstand structure ,naming cells default from 0,1,2")
                cell_image.save(save_dir + "/" + f"{prefix}_cell_{i}" + ".jpg")

        logger.info("cutting cells done")


def draw_locate(
    image: Union[str, Image.Image],
    bboxs: list,
    save_dir,
    cut=False,
    return_crops=False,
    adjust_ratio=0.05,
):
    if isinstance(image, str):
        image = img_load_by_Image(image).convert("RGB")
    table_locate_draw_image = image.copy()
    locate_draw = ImageDraw.Draw(table_locate_draw_image)
    # bboxs = sorted(bboxs, key=lambda x: (x[2] + x[0]) / 2 + (x[3] + x[1]) / 2)
    crop_tables = []
    for i, bbox in enumerate(bboxs):
        adjust_width = (bbox[2] - bbox[0]) * adjust_ratio * 0.5
        adjust_height = (bbox[3] - bbox[1]) * adjust_ratio * 0.5

        bbox[0] -= adjust_width
        bbox[1] -= adjust_height
        bbox[2] += adjust_width
        bbox[3] += adjust_height

        bbox = [int(round(float(xy))) for xy in bbox]
        crop_box = bbox
        draw_box = [(bbox[0], bbox[1]), (bbox[2], bbox[3])]
        if cut:
            crop_table = image.crop(crop_box)
            crop_table.save(save_dir + "/" + f"locate_table_{i+1}.jpg")
            if return_crops:
                crop_tables.append(crop_table)
        locate_draw.rectangle(draw_box, outline="red", width=3)
    table_locate_draw_image.save(save_dir + "/" + "locate_table.jpg")
    if return_crops:
        return crop_tables


def adjst_box(bbox: list, ratio=0.05, enlarge=True, toint=True):
    adjust_width = (bbox[2] - bbox[0]) * ratio * 0.5
    adjust_height = (bbox[3] - bbox[1]) * ratio * 0.5
    if enlarge:
        bbox[0] -= adjust_width
        bbox[1] -= adjust_height
        bbox[2] += adjust_width
        bbox[3] += adjust_height
    else:
        bbox[0] += adjust_width
        bbox[1] += adjust_height
        bbox[2] -= adjust_width
        bbox[3] -= adjust_height
    if toint:
        bbox = [int(round(float(xy))) for xy in bbox]
    return bbox


def evaluate_valid_cells(cells):
    # 转换为numpy数组并计算基础统计量
    cells_np = np.array(cells)
    widths = cells_np[:, 2] - cells_np[:, 0]
    heights = cells_np[:, 3] - cells_np[:, 1]

    # 计算统计指标
    stats = {
        "mean_w": np.mean(widths),
        "mean_h": np.mean(heights),
        "std_w": np.std(widths),
        "std_h": np.std(heights),
        "total": len(cells),
    }

    # 定义标准差范围评估函数
    def calculate_valid(n_std):
        # 计算有效范围边界
        w_low = stats["mean_w"] - n_std * stats["std_w"]
        w_high = stats["mean_w"] + n_std * stats["std_w"]
        h_low = stats["mean_h"] - n_std * stats["std_h"]
        h_high = stats["mean_h"] + n_std * stats["std_h"]

        # 创建布尔掩码
        width_mask = (widths >= w_low) & (widths <= w_high)
        height_mask = (heights >= h_low) & (heights <= h_high)
        valid_mask = width_mask & height_mask  # 需要同时满足宽高条件

        # 返回统计结果
        return {
            "valid_count": np.sum(valid_mask),
            "valid_ratio": np.mean(valid_mask),
            "valid_indices": np.where(valid_mask)[0].tolist(),
            "thresholds": {
                "width": (float(w_low), float(w_high)),
                "height": (float(h_low), float(h_high)),
            },
        }

    # 计算不同标准差范围的结果
    return {
        "0.5_std": calculate_valid(0.5),
        "1_std": calculate_valid(1),
        "1.5_std": calculate_valid(1.5),
        "2_std": calculate_valid(2),
        "stats": stats,
    }


def filter_by_w_h(
    cells, img_shape: tuple, w_range: tuple, h_range: tuple, confs: list = None
):
    if len(cells) == 0:
        return [], []
    logger.info(f"before filtering, total {len(cells)} cells")

    state = evaluate_valid_cells(cells)
    for key, val in state.items():
        logger.info(f"{key} : {val}")

    w, h = img_shape  # Image size
    logger.info(f"img_shape: w={w} , h={h}")
    min_w_ratio, max_w_ratio = w_range
    min_h_ratio, max_h_ratio = h_range
    min_w, max_w = min_w_ratio * w, max_w_ratio * w
    min_h, max_h = min_h_ratio * h, max_h_ratio * h
    logger.info(f"w_range: {min_w} ~ {max_w}")
    logger.info(f"h_range: {min_h} ~ {max_h}")
    filtered_cells = []
    filtered_confs = []
    filtered_idx = []
    for i, cell in enumerate(cells):
        cell_w = cell[2] - cell[0]
        cell_h = cell[3] - cell[1]
        # 限制最小 w/h
        if cell_h < min_h and (cell_w < min_w or cell_w > max_w):
            filtered_idx.append(i)
            continue

        elif cell_w < min_w and (cell_h < min_h or cell_h > max_h):
            filtered_idx.append(i)
            continue

        # if cell_w < min_w or cell_w > max_w or cell_h < min_h or cell_h > max_h:
        #     filtered_idx.append(i)
        #     continue

        filtered_cells.append(cell)
        if confs is not None:
            filtered_confs.append(confs[i])
    logger.info(f"filtering {len(filtered_idx)} cells, the idxs: {filtered_idx}")
    logger.info(f"remaing {len(filtered_cells)} cells")
    return filtered_cells, filtered_confs


def filter_by_mean_std(cells, n_std):
    if n_std not in [0.5, 1, 1.5, 2]:
        raise ValueError(f"n_std 参数只能为 {[0.5, 1, 1.5, 2]}")

    if not isinstance(cells, (list, np.ndarray)):
        raise TypeError("cells 应为列表或 numpy 数组")
    evaluation_result = evaluate_valid_cells(cells)
    np_cells = np.array(cells)
    std_key = f"{n_std}_std"
    valid_indices = evaluation_result[std_key]["valid_indices"]
    if isinstance(cells, np.ndarray):
        # numpy 数组直接索引
        return cells[valid_indices]
    else:
        # 列表推导式处理
        return [cells[i] for i in valid_indices]


def visualize_filtered(cells, filtered_idx, img_size=(800, 600), show=True):
    """可视化过滤效果"""
    img = np.ones((*img_size, 3)) * 255  # 创建白色背景

    # 绘制所有原始框（灰色）
    for x1, y1, x2, y2 in cells:
        cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 200), 1)

    # 高亮有效框（绿色）
    for i in filtered_idx:
        x1, y1, x2, y2 = cells[i]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    plt.imshow(img)
    plt.title(f"Valid Cells ({len(filtered_idx)}/{len(cells)})")
    plt.savefig("visualize_filtered.jpg")
    if show:
        plt.show()


import numpy as np
from scipy.spatial import KDTree  # 用于加速最近邻搜索


def find_nearest_boxes(source_boxes, target_boxes, distance_type="center", k=1):
    """
    为每个源框找到目标框集合中最近的k个框

    参数：
    source_boxes -- 源框列表，格式 [[x1,y1,x2,y2], ...] 或 numpy数组
    target_boxes -- 目标框列表，格式同上
    distance_type -- 距离计算方式：
        'center' : 中心点欧氏距离（默认）
        'iou'    : 交并比（此时找最大iou而非最小距离）
        'edge'   : 边缘最近距离
    k -- 返回最近邻的数量

    返回：
    nearest_info -- 包含最近邻索引和距离的列表，每个元素为 (indices, distances)
    """
    # 转换为 numpy 数组
    src = np.asarray(source_boxes)
    tgt = np.asarray(target_boxes)

    # 空数据校验
    if len(src) == 0 or len(tgt) == 0:
        return []

    # 根据距离类型选择计算方法
    if distance_type == "center":
        # 计算中心点坐标
        src_centers = (src[:, :2] + src[:, 2:]) / 2
        tgt_centers = (tgt[:, :2] + tgt[:, 2:]) / 2

        # 构建KDTree加速搜索
        tree = KDTree(tgt_centers)
        distances, indices = tree.query(src_centers, k=k)

    elif distance_type == "iou":
        # 计算交并比（找最大IoU）
        iou_matrix = pairwise_iou(src, tgt)
        indices = np.argpartition(iou_matrix, -k, axis=1)[:, -k:]
        distances = np.take_along_axis(iou_matrix, indices, axis=1)
        # 转换为距离形式（1 - IoU）
        distances = 1 - distances

    elif distance_type == "edge":
        # 计算边缘最近距离
        dist_matrix = edge_distance_matrix(src, tgt)
        indices = np.argpartition(dist_matrix, k, axis=1)[:, :k]
        distances = np.take_along_axis(dist_matrix, indices, axis=1)

    else:
        raise ValueError("不支持的 distance_type，可选：'center', 'iou', 'edge'")

    # 转换为友好格式
    return [(idx.tolist(), dist.tolist()) for idx, dist in zip(indices, distances)]


def pairwise_iou(boxes1, boxes2):
    """计算两组框之间的 IoU 矩阵"""
    # 计算交集区域
    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = np.clip(rb - lt, 0, None)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    # 计算各自面积
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # [N]
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # [M]

    # 计算并集
    union = area1[:, None] + area2 - inter  # [N,M]

    return inter / (union + 1e-8)  # 防止除以零


def edge_distance_matrix(boxes1, boxes2):
    """计算两组框之间的边缘最小距离矩阵"""
    # 扩展维度用于广播计算 [N,1,4] vs [1,M,4]
    b1 = boxes1[:, None, :]
    b2 = boxes2[None, :, :]

    # 计算水平方向距离
    left = b2[..., 2] - b1[..., 0]  # box2右 - box1左
    right = b1[..., 2] - b2[..., 0]  # box1右 - box2左
    horizontal = np.maximum(left, right)

    # 计算垂直方向距离
    top = b2[..., 3] - b1[..., 1]  # box2下 - box1上
    bottom = b1[..., 3] - b2[..., 1]  # box1下 - box2上
    vertical = np.maximum(top, bottom)

    # 组合成欧氏距离
    return np.sqrt(np.maximum(horizontal, 0) ** 2 + np.maximum(vertical, 0) ** 2)


def visualize_box_connections(
    source_boxes,
    target_boxes,
    connections,
    image=None,
    canvas_size=None,
    line_style="dashed",
    save_path=None,
    show=False,
    dpi=150,
):
    """
    可视化两组框及其连接关系（支持在现有图像上绘制）

    参数：
    source_boxes  -- 源框列表，格式 [[x1,y1,x2,y2], ...] 或 numpy数组
    target_boxes  -- 目标框列表，格式同上
    connections   -- 连接关系，格式 [(src_idx, tgt_idx), ...] 或 [[src_idx, tgt_idx], ...]
    image         -- 背景图像，支持 numpy数组 或 PIL.Image，None时为空白画布
    canvas_size   -- 画布尺寸 (width, height)，当image存在时自动忽略
    line_style    -- 连接线样式：'dashed', 'solid', 'dotted' 等
    save_path     -- 图片保存路径，None时不保存
    show          -- 是否显示可视化结果
    dpi           -- 输出图像分辨率
    """
    # 转换为 numpy 数组
    src = np.asarray(source_boxes)
    tgt = np.asarray(target_boxes)

    # 创建画布
    fig = plt.figure(dpi=dpi)
    ax = fig.add_subplot(111)

    # 背景图像处理
    if image is not None:
        # 统一转换为RGB numpy数组
        if isinstance(image, np.ndarray):
            if image.ndim == 2:  # 灰度图转RGB
                img_array = np.stack([image] * 3, axis=-1)
            elif image.ndim == 3:
                if image.shape[2] == 4:  # 去除alpha通道
                    img_array = image[:, :, :3]
                else:
                    img_array = image.copy()
            # OpenCV BGR转RGB
            if img_array.dtype == "uint8" and np.array_equal(
                img_array[..., 0], image[..., -1]
            ):
                img_array = img_array[..., ::-1]
        elif isinstance(image, Image.Image):
            img_array = np.array(image.convert("RGB"))
        else:
            raise TypeError("不支持的图像格式，请使用numpy数组或PIL.Image")

        # 显示图像并获取实际尺寸
        ax.imshow(img_array)
        canvas_width = img_array.shape[1]
        canvas_height = img_array.shape[0]
    else:
        # 自动计算画布尺寸
        if canvas_size is None:
            all_x = np.concatenate([src[:, [0, 2]], tgt[:, [0, 2]]])
            all_y = np.concatenate([src[:, [1, 3]], tgt[:, [1, 3]]])
            canvas_width = int(np.max(all_x) + 10) if len(all_x) > 0 else 800
            canvas_height = int(np.max(all_y) + 10) if len(all_y) > 0 else 600
        else:
            canvas_width, canvas_height = canvas_size

        ax.set_xlim(0, canvas_width)
        ax.set_ylim(canvas_height, 0)  # 图像坐标系

    ax.set_aspect("equal")

    # 绘制源框（红色实线带透明填充）
    for box in src:
        x1, y1, x2, y2 = box
        rect = plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=True,
            color="red",
            alpha=0.2,
            edgecolor="darkred",
            linewidth=2,
            label="Source",
        )
        ax.add_patch(rect)

    # 绘制目标框（蓝色实线带透明填充）
    for box in tgt:
        x1, y1, x2, y2 = box
        rect = plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=True,
            color="blue",
            alpha=0.2,
            edgecolor="darkblue",
            linewidth=2,
            label="Target",
        )
        ax.add_patch(rect)

    # 绘制连接线（绿色虚线）
    seen_pairs = set()
    for pair in connections:
        if len(pair) < 2:
            continue
        src_idx, tgt_idx = pair[0], pair[1]

        if src_idx >= len(src) or tgt_idx >= len(tgt):
            continue

        src_box = src[src_idx]
        tgt_box = tgt[tgt_idx]
        src_center = [(src_box[0] + src_box[2]) / 2, (src_box[1] + src_box[3]) / 2]
        tgt_center = [(tgt_box[0] + tgt_box[2]) / 2, (tgt_box[1] + tgt_box[3]) / 2]

        # 动态调整线宽和透明度
        distance = np.hypot(
            src_center[0] - tgt_center[0], src_center[1] - tgt_center[1]
        )
        linewidth = max(0.5, 2 - distance / 100)
        alpha = max(0.3, 1 - distance / 500)

        if (src_idx, tgt_idx) not in seen_pairs:
            ax.plot(
                [src_center[0], tgt_center[0]],
                [src_center[1], tgt_center[1]],
                color="lime",
                linestyle=line_style,
                linewidth=linewidth,
                alpha=alpha,
                label="Connection" if not seen_pairs else "",
            )
            seen_pairs.add((src_idx, tgt_idx))
        else:
            ax.plot(
                [src_center[0], tgt_center[0]],
                [src_center[1], tgt_center[1]],
                color="lime",
                linestyle=line_style,
                linewidth=linewidth,
                alpha=alpha,
            )

    # 图例处理
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    for h, l in zip(handles, labels):
        if l not in unique_labels:
            unique_labels.append(l)
            unique_handles.append(h)
    ax.legend(unique_handles, unique_labels, loc="upper right")

    plt.title(f"Box Connections Visualization ({len(connections)} pairs)")
    plt.axis("off")  # 隐藏坐标轴

    # 保存与显示
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=dpi)
    if show:
        plt.show()
    plt.close()


# # 示例用法 ##############################################
# if __name__ == "__main__":
#     # 生成测试数据
#     np.random.seed(42)
#     source_boxes = np.random.randint(0, 100, (5, 4))
#     target_boxes = np.random.randint(50, 150, (10, 4))

#     # 查找最近邻
#     nearest_info = find_nearest_boxes(source_boxes, target_boxes, k=2)

#     # 可视化结果
#     for i, (indices, dists) in enumerate(nearest_info):
#         print(f"源框 {i} [{
#             ','.join(map(str, source_boxes[i]))}]")
#         for j, idx in enumerate(indices):
#             print(f"  第{j+1}近目标框：索引 {idx}，距离 {dists[j]:.2f}，坐标 {target_boxes[idx]}")
