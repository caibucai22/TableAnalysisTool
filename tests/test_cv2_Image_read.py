import cv2
from PIL import Image
import numpy as np

# img_path = "C:/Users/001/Pictures/ocr/v2/微信图片_20250117213130.jpg"
img_path = "C:/Users/001/Pictures/ocr/v2/微信图片_20250117213134.jpg"
# img_path = "C:/Users/001/Pictures/ocr/v2/微信图片_20250217205055.jpg"


cv2_img = cv2.imdecode(
    np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR
)  # support chinese
# print(cv2_img.size)
# cv2_img = cv2.imread(img_path)
print(cv2_img.shape)  # h w c

Image_img = Image.open(img_path)
print(Image_img.size)  # w h
exif = Image_img.getexif()
orientation = exif.get(0x0112)
print(orientation)

# default short->w
