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

import os
import subprocess
import platform
import cv2
import numpy as np


def open_folder(folder_path):
    if platform.system() == "Windows":
        os.startfile(folder_path)
    elif platform.system() == "Darwin":  # macOS
        subprocess.Popen(["open", folder_path])
    else:  # Linux and others
        subprocess.Popen(["xdg-open", folder_path])


def find_bingo_number(num_list):
    """
    查找 bingo 对应的数字。

    Args:
        num_list: 包含数字和 "bingo" 的字符串列表。

    Returns:
        如果找到 "bingo"，则返回对应的数字（1-4），否则返回 None。
        如果输入列表格式错误（例如包含非数字或非 "bingo" 的元素），则抛出 ValueError。
    """
    try:
        numbers = [int(num) for num in num_list]
    except ValueError as e:
        # 查找 "bingo" 的索引
        try:
            bingo_index = num_list.index("bingo")
        except ValueError:
            raise ValueError("列表格式错误：包含非数字或非 'bingo' 的元素") from e

        # 根据索引推断 bingo 对应的数字
        if bingo_index < len(num_list):
            numbers = [int(num) for num in num_list if num !=
                       "bingo"]  # 在except中进行赋值，避免未赋值错误
            missing_numbers = set(range(1, 5)) - set(numbers)
            if len(missing_numbers) == 1:
                return missing_numbers.pop()
            else:
                raise ValueError("无法确定 bingo 对应的数字，缺失数字数量不为1")
        else:
            raise ValueError("bingo 位置超出列表长度")

    return None  # 没有 bingo


def binarize_images(folder_path, threshold=127, max_value=255, method=cv2.THRESH_BINARY, adaptive_method=None, block_size=11, C=2):
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
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):  # 检查文件是否为图像
            image_path = os.path.join(folder_path, filename)
            try:
                img = cv2.imread(image_path)
                if img is None:
                    print(f"警告：无法读取图像 '{filename}'。请检查文件是否损坏。")
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                if adaptive_method is not None:  # 使用自适应阈值
                    binary = cv2.adaptiveThreshold(
                        gray, max_value, adaptive_method, cv2.THRESH_BINARY, block_size, C)
                elif method == cv2.THRESH_OTSU:  # 使用otsu算法
                    _, binary = cv2.threshold(
                        gray, 0, max_value, method | cv2.THRESH_OTSU)
                else:  # 使用全局阈值
                    _, binary = cv2.threshold(
                        gray, threshold, max_value, method)
                
                # 滤波
                binary = cv2.medianBlur(binary, 3)

                output_filename = "binary_" + filename  # 输出文件名
                output_path = os.path.join(folder_path, output_filename)
                cv2.imwrite(output_path, binary)
                print(f"{filename} binarized, save to {output_filename}",end=' ')

                total_pixels = binary.size
                white_pixels = cv2.countNonZero(binary)  # 更高效的白色像素计数
                black_pixels = total_pixels - white_pixels

                white_percentage = (white_pixels / total_pixels) * 100
                black_percentage = (black_pixels / total_pixels) * 100
                black_percentage_list.append(black_percentage)
                print('black:white = ', white_percentage, ':', black_percentage)

            except Exception as e:
                print(f"处理图像 '{filename}' 时发生错误：{e}")
    mean = np.array(black_percentage_list).mean()
    std = np.array(black_percentage_list).std()
    print('mean: ',mean, ' std: ',std)
    indices_point5std = [i for i, value in enumerate(black_percentage_list) if abs(value - mean) > 0.5*std]
    indices_1std = [i for i, value in enumerate(black_percentage_list) if abs(value - mean) > std]

    print('indices_point5std: ',np.array(filenames)[indices_point5std])
    print('indices_1std: ',np.array(filenames)[indices_1std])
