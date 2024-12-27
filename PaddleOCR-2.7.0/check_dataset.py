import cv2
import os, json
import numpy as np
from matplotlib import pyplot as plt

def parse_line(data_dir, line):
    data_line = line.strip("\n")
    info = json.loads(data_line)
    file_name = info['filename']
    cells = info['html']['cells'].copy()
    structure = info['html']['structure']['tokens'].copy()

    img_path = os.path.join(data_dir, file_name)
    if not os.path.exists(img_path):
        print(img_path)
        return None
    data = {
        'img_path': img_path,
        'cells': cells,
        'structure': structure,
        'file_name': file_name
    }
    return data

def draw_bbox(img_path, points, color=(255, 0, 0), thickness=2):
    if isinstance(img_path, str):
        img_path = cv2.imread(img_path)
    img_path = img_path.copy()
    for point in points:
        cv2.polylines(img_path, [point.astype(int)], True, color, thickness)
    return img_path


def rebuild_html(data):
    html_code = data['structure']
    cells = data['cells']
    to_insert = [i for i, tag in enumerate(html_code) if tag in ('<td>', '>')]

    for i, cell in zip(to_insert[::-1], cells[::-1]):
        if cell['tokens']:
            text = ''.join(cell['tokens'])
            # skip empty text
            sp_char_list = ['<b>', '</b>', '\u2028', ' ', '<i>', '</i>']
            text_remove_style = skip_char(text, sp_char_list)
            if len(text_remove_style) == 0:
                continue
            html_code.insert(i + 1, text)

    html_code = ''.join(html_code)
    return html_code


def skip_char(text, sp_char_list):
    """
    skip empty cell
    @param text: text in cell
    @param sp_char_list: style char and special code
    @return:
    """
    for sp_char in sp_char_list:
        text = text.replace(sp_char, '')
    return text

save_dir = '/home/aistudio/vis'
os.makedirs(save_dir, exist_ok=True)
image_dir = '/home/aistudio/data/data165849/'
html_str = '<table border="1">'

# 解析标注信息并还原html表格
data = parse_line(image_dir, val_list[0])

img = cv2.imread(data['img_path'])
img_name = ''.join(os.path.basename(data['file_name']).split('.')[:-1])
img_save_name = os.path.join(save_dir, img_name)
boxes = [np.array(x['bbox']) for x in data['cells']]
show_img = draw_bbox(data['img_path'], boxes)
cv2.imwrite(img_save_name + '_show.jpg', show_img)

html = rebuild_html(data)
html_str += html
html_str += '</table>'

# 显示单元格坐标
plt.figure(figsize=(15,15))
plt.imshow(show_img)
plt.show()