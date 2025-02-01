import datetime
import logging
import logging.handlers
import os

import termcolor
from PyQt5.QtCore import QObject, pyqtSignal

COLORS = {
    "WARNING": "yellow",
    "INFO": "white",
    "DEBUG": "blue",
    "CRITICAL": "red",
    "ERROR": "red",
}


class HTMLFormatter(logging.Formatter):
    """将彩色终端代码转换为HTML格式"""

    color_map = {
        "red": "#ff0000",
        "green": "#00ff00",
        "yellow": "#ffff00",
        "blue": "#0000ff",
        "cyan": "#00ffff",
        "white": "#ffffff",
    }

    def __init__(self, fmt):
        logging.Formatter.__init__(self, fmt, datefmt="%Y/%m/%d %H:%M:%S")

    def format(self, record):
        message = super().format(record)
        return self.convert_colors(message)

    def convert_colors(self, text):
        for termcolor_, htmlcolor in self.color_map.items():
            text = text.replace(
                termcolor.colored("", color=termcolor_, attrs={"bold": True}),
                f'<span style="color:{htmlcolor}; font-weight:bold">',
            )
        return text + "</span>"


class ColoredFormatter(logging.Formatter):
    def __init__(self, fmt, use_color=True):
        logging.Formatter.__init__(self, fmt, datefmt="%Y/%m/%d %H:%M:%S")
        self.use_color = use_color

    def format(self, record):
        levelname = record.levelname
        if self.use_color and levelname in COLORS:

            def colored(text):
                return termcolor.colored(
                    text,
                    color=COLORS[levelname],
                    attrs={"bold": True},
                )

            record.levelname2 = colored(f"{record.levelname:<7}")
            record.message2 = colored(record.msg)

            asctime2 = datetime.datetime.fromtimestamp(record.created)
            record.asctime2 = termcolor.colored(asctime2, color="green")

            record.module2 = termcolor.colored(record.module, color="cyan")
            record.funcName2 = termcolor.colored(record.funcName, color="cyan")
            record.lineno2 = termcolor.colored(record.lineno, color="cyan")
        return logging.Formatter.format(self, record)


class ColoredLogger(logging.Logger):
    # FORMAT = "[%(levelname2)s] %(module2)s:%(funcName2)s:%(lineno2)s - %(message2)s"
    FORMAT = ("[%(asctime)s] %(levelname)s: %(filename)s <%(funcName)s> %(message)s",)

    def __init__(self, name):
        logging.Logger.__init__(self, name, logging.INFO)

        color_formatter = ColoredFormatter(self.FORMAT)

        console = logging.StreamHandler()
        console.setFormatter(color_formatter)

        self.addHandler(console)


class GUILogHandler(QObject, logging.Handler):
    log_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        logging.Handler.__init__(self)
        self.setFormatter(HTMLFormatter("[%(asctime)s] %(levelname)s: %(message)s"))

    def emit(self, record):
        msg = self.format(record)
        # self.log_signal.emit(msg)
        self.log_signal.emit(f"<pre>{msg}</pre>")


class TripleLogger(logging.Logger):
    def __init__(self, name):
        super().__init__(name, logging.INFO)

        # GUI处理器（HTML格式）
        self.gui_handler = GUILogHandler()

        # 控制台处理器（彩色）
        console_formatter = ColoredFormatter(
            "[%(asctime)s] %(levelname)s: %(filename)s <%(funcName)s> %(message)s"
        )
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        self.addHandler(console_handler)

        # 文件处理器（纯文本）
        file_handler = logging.handlers.RotatingFileHandler("log.txt")
        file_formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s: %(filename)s <%(funcName)s> %(message)s",
            datefmt="%Y/%m/%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        self.addHandler(file_handler)

    def add_gui_handler(self, widget):
        self.addHandler(self.gui_handler)
        self.gui_handler.log_signal.connect(widget.append)


# logger = logging.getLogger("ImageProcessAPP")
# logger.__class__ = ColoredLogger
# logger.setLevel(logging.INFO)

logger = TripleLogger("ImageProcessAPPv2")
logger.setLevel(logging.INFO)
