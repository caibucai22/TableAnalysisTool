# -*- coding: UTF-8 -*-
"""
@File    ：app.py
@Author  ：Csy
@Date    ：2024/12/24 16:58 
@Bref    :
@Ref     :
TODO     :
         :
"""
import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
from app_ui import ImageProcessingApp

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageProcessingApp()
    window.show()
    QTimer.singleShot(500,window.show_instructions)
    sys.exit(app.exec_())
