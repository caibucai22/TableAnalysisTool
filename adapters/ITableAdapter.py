# -*- coding: UTF-8 -*-
"""
@File    ：ITableAdapter.py
@Author  ：Csy
@Date    ：2025/01/08 14:10 
@Bref    : 适配器模式 模型表格结构+ocr输出统一到Table实体中
@Ref     : # https://geek-docs.com/python/python-ask-answer/219_python_how_do_i_implement_interfaces_in_python.html
TODO     :
         :
"""
from abc import ABC, abstractmethod


class ITableAdapter(ABC):

    @abstractmethod
    def adapt(self, model_output, **kwargs):
        pass
