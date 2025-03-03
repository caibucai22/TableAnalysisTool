import cv2
import numpy as np


def detect_lines_hough(image_path):
    """
    使用霍夫线变换检测图像中的直线（包括横线和竖线）。

    Args:
        image_path: 图像文件路径。

    Returns:
        img_with_lines: 绘制了检测到的直线的图像 (NumPy 数组)。
    """
    # 1. 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误: 无法读取图像文件: {image_path}")
        return None

    # 2. 转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. 边缘检测 (Canny 边缘检测)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)  # 可以调整 Canny 阈值 (50, 150)

    # 4. 霍夫线变换
    #   - rho:  累加器分辨率 (像素单位)
    #   - theta: 角度分辨率 (弧度单位)
    #   - threshold: 累加器阈值，只有累加计数超过阈值的直线才被检测出来
    lines = cv2.HoughLines(
        edges, 1, np.pi / 180, 100
    )  # 可以调整 HoughLines 参数 (1, np.pi/180, 100)

    # 5. 在原图上绘制检测到的直线 (区分横线和竖线)
    img_with_lines = img.copy()  # 复制一份原图，在上面绘制直线
    horizontal_lines = []
    vertical_lines = []

    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            # 判断直线是横线还是竖线 (根据角度 theta)
            if (
                abs(theta) < np.pi / 8 or abs(theta - np.pi) < np.pi / 8
            ):  #  角度接近 0 或 180 度 (水平线，允许一定角度偏差)
                horizontal_lines.append(((x1, y1), (x2, y2)))
                cv2.line(
                    img_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 2
                )  # 红色 - 横线
            elif (
                abs(theta - np.pi / 2) < np.pi / 8 or abs(theta + np.pi / 2) < np.pi / 8
            ):  # 角度接近 90 或 270 度 (竖直线，允许一定角度偏差)
                vertical_lines.append(((x1, y1), (x2, y2)))
                cv2.line(
                    img_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2
                )  # 绿色 - 竖线

    print(f"检测到横线: {len(horizontal_lines)} 条")
    print(f"检测到竖线: {len(vertical_lines)} 条")

    # 6. 显示结果 (可选)
    cv2.imshow("Original Image", img)
    cv2.imshow("Edge Image", edges)
    cv2.imshow("Lines Detected Image", img_with_lines)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return img_with_lines


if __name__ == "__main__":
    image_file = "tests/cells/cell_1_2.jpg"  # 替换成你的图像文件路径
    output_image = detect_lines_hough(image_file)
    if output_image is not None:
        cv2.imwrite("lines_detected.jpg", output_image)  # 保存结果图像
        print("直线检测结果已保存到 lines_detected.png")
