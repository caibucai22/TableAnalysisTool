import os
import cv2
import numpy as np
import openvino
from PIL import Image


class Yolov8_cls_CV:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            print("Model Path not exist!")
            exit(-1)

        self.m_ov_core = openvino.Core()
        self.m_ov_model = self.m_ov_core.read_model(model=model_path)
        self.m_ov_compiled_model = self.m_ov_core.compile_model(
            model=self.m_ov_model, device_name="AUTO")
        self.m_ov_infer_request = self.m_ov_compiled_model.create_infer_request()

    def __call__(self, image_path: str):
        if not os.path.exists(image_path):
            print("Image Path not exist!")
            return

        image = cv2.imread(image_path)

        return self.Infer(image=image)

    def Infer(self, image: cv2.Mat):
        input_tensor = self.Preprocessing(image=image)
        output_tensor = self.__Internal_Infer__(tensor=input_tensor)
        return self.Postprocessing(tensor=output_tensor)

    # 不是cv2Color的问题， 这个表象他只是显示不同，但是他的值是一致的

    def Preprocessing(self, image: cv2.Mat):

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.Resize(image=image)

        image = self.CenterCrop(image=image)

        return np.expand_dims(np.transpose(image, (2, 0, 1)), axis=0).astype('float32') / 255.0

    def Postprocessing(self, tensor: openvino.Tensor):
        conf_array = tensor.data[0]
        indexed_array = [(index, value)
                         for index, value in enumerate(conf_array)]
        sorted_array = sorted(indexed_array, key=lambda x: x[1], reverse=True)

        top5_values = [x[1] for x in sorted_array[:5]]
        top5_indices = [x[0] for x in sorted_array[:5]]

        return top5_values, top5_indices

    def __Internal_Infer__(self, tensor: openvino.Tensor) -> openvino.Tensor:
        self.m_ov_infer_request.infer(inputs=tensor)
        return self.m_ov_infer_request.get_output_tensor()

    def CenterCrop(self, image: cv2.Mat, size=[224, 224]):

        image_height, image_width = image.shape[:2]
        crop_height, crop_width = size

        crop_top = int(round((image_height - crop_height) / 2.0))
        crop_left = int(round((image_width - crop_width) / 2.0))
        return image[crop_top:crop_top+crop_height, crop_left:crop_left+crop_width]

    def Resize(self, image: cv2.Mat, size=[224]):

        image_size = image.shape[:2]

        if len(size) == 1:  # specified size only for the smallest edge
            h, w = image_size
            short, long = (w, h) if w <= h else (h, w)
            requested_new_short = size if isinstance(size, int) else size[0]

            new_short, new_long = requested_new_short, int(
                requested_new_short * long / short)

            new_w, new_h = (new_short, new_long) if w <= h else (
                new_long, new_short)
        else:  # specified both h and w
            new_w, new_h = size[1], size[0]

        return cv2.resize(src=image, dsize=(new_w, new_h), interpolation=cv2.INTER_LINEAR)


class Yolov8_cls_PIL:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            print("Model Path not exist!")
            exit(-1)

        self.m_ov_core = openvino.Core()
        self.m_ov_model = self.m_ov_core.read_model(model=model_path)
        self.m_ov_compiled_model = self.m_ov_core.compile_model(
            model=self.m_ov_model, device_name="AUTO")
        self.m_ov_infer_request = self.m_ov_compiled_model.create_infer_request()

    def __call__(self, image: Image.Image):
        # def __call__(self, image_path: str):
        # if not os.path.exists(image_path):
        #     print("Image Path not exist!")
        #     return

        # image = Image.open(image_path)

        return self.Infer(image=image)

    def Infer(self, image: Image.Image):
        input_tensor = self.Preprocessing(image=image)
        output_tensor = self.__Internal_Infer__(tensor=input_tensor)
        return self.Postprocessing(tensor=output_tensor)

    def Preprocessing(self, image: Image.Image):

        image = image.convert("RGB")
        # image.save("PIL_RGB.jpg")

        image = self.Resize(image=image)
        # image.save("PIL_Resized.jpg")

        image = self.CenterCrop(image=image)
        # image.save("PIL_Croped.jpg")

        npimage = np.array(image)

        return np.expand_dims(npimage.transpose(2, 0, 1), axis=0).astype(np.float32) / 255.0

    def Postprocessing(self, tensor: openvino.Tensor):
        conf_array = tensor.data[0]
        indexed_array = [(index, value)
                         for index, value in enumerate(conf_array)]
        sorted_array = sorted(indexed_array, key=lambda x: x[1], reverse=True)

        top5_values = [x[1] for x in sorted_array[:5]]
        top5_indices = [x[0] for x in sorted_array[:5]]

        return top5_values, top5_indices

    def __Internal_Infer__(self, tensor: openvino.Tensor) -> openvino.Tensor:
        self.m_ov_infer_request.infer(inputs=tensor)
        return self.m_ov_infer_request.get_output_tensor()

    def CenterCrop(self, image: Image.Image, size=[224, 224]):

        image_width, image_height = image.size
        crop_height, crop_width = size

        crop_top = int(round((image_height - crop_height) / 2.0))
        crop_left = int(round((image_width - crop_width) / 2.0))

        return image.crop((crop_left, crop_top, crop_left+crop_width, crop_top+crop_height))

    def Resize(self, image: Image.Image, size=[224]):
        "保持宽高比不变进行resize, 以短的为基础"
        image_size = image.size[::-1]

        # 默认是224x224
        if len(size) == 1:
            h, w = image_size
            short, long = (w, h) if w <= h else (h, w)
            requested_new_short = size if isinstance(size, int) else size[0]

            new_short, new_long = requested_new_short, int(
                requested_new_short * long / short)

            new_w, new_h = (new_short, new_long) if w <= h else (
                new_long, new_short)
        else:  # specified both h and w
            new_w, new_h = size[1], size[0]

        return image.resize((new_w, new_h), resample=Image.BILINEAR)


if __name__ == "__main__":
    clser = Yolov8_cls_PIL(
        "E:/Models/ultralytics-8.1.0/train_bingo/bingo-cls/n-base/weights/bingo-cls.onnx")
    result = clser(
        "E:/01-LabProjects-Data/BingoDataset/BingoDataset-cls/train/bingo/cell_6_2.jpg")

    top5_conf = result[0]
    top5_index = result[1]
    for idx in range(len(top5_conf)):
        print("index:{}, conf:{:.2f}".format(top5_index[idx], top5_conf[idx]))
