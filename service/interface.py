from abc import ABC, abstractmethod
from typing import Any, Type
import numpy as np
from adapters.Table import TableEntity


class ITableLocateService(ABC):
    @abstractmethod
    def locate_table(self, image_path: str) -> np.ndarray:
        """输入图片路径，返回裁剪后的表格区域图像（BGR格式）"""
        pass


class ITableStructureService(ABC):
    @abstractmethod
    def recognize_structure(self, table_image: np.ndarray) -> TableEntity:
        """输入表格图像，返回结构化数据（如 TableEntity 对象）"""
        pass


class IOCRService(ABC):
    @abstractmethod
    def recognize_text(self, cell_images: list) -> list:
        """输入单元格图像列表，返回文本列表"""
        pass


class IClsService(ABC):
    @abstractmethod
    def binary_cls(self, cell_images: list) -> list:
        """输入单元格图像列表，返回文本列表"""
        pass


class ICustomService(ABC):
    @abstractmethod
    def service(self, images: list):
        pass
