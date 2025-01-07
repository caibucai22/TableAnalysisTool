import logging
import logging.handlers
import sys
import os
import functools

logger_initialized = {}
logging.basicConfig(level=logging.DEBUG)
# DEBUG  INFO WARNING ERROR CRITICAL


@functools.lru_cache
def get_logger(name='TableAnalysisTool', log_file='./log.txt', log_level=logging.DEBUG):
    logger = logging.getLogger(name)

    if name in logger_initialized:
        return logger_initialized[name]
    # for logger_name in logger_initialized:
    #     if name.startswith(logger_name):
    #         return logger
    formatter = logging.Formatter('[%(asctime)s] %(filename)s [line:%(lineno)d] %(levelname)s: %(funcName)s %(message)s',
                                  datefmt="%Y/%m/%d %H:%M:%S")
    if not logger.handlers:
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if log_file is not None:
            log_file_folder = os.path.split(log_file)[0]
            os.makedirs(log_file_folder, exist_ok=True)
            # file_handler = logging.FileHandler(log_file, 'a') # repeat TimedRotatingFileHandler
            # file_handler.setFormatter(formatter)
            # logger.addHandler(file_handler)
        # file size control
        filesize_handler = logging.handlers.TimedRotatingFileHandler(
            log_file, when='midnight', backupCount=5)
        filesize_handler.setFormatter(formatter)
        logger.addHandler(filesize_handler)

    logger.setLevel(log_level)
    logger_initialized[name] = logger
    logger.propagate = False
    return logger


if __name__ == "__main__":

    logger = get_logger(log_file='./log.txt', log_level=logging.DEBUG)
    print(logger.handlers)
    logger.info("info ")
    print(logger.handlers)
    logger.error("error ")
    logger.error("test catch exception", exc_info=True)
    logger.debug('test debug')

    # logger2 = get_logger(name='TableProcess',
    #                      log_file='./log.txt', log_level=logging.INFO)
    # logger2.info("info 1")
    # logger2.info("info 2")
    # logger2.error("error ")
    # try:
    #     raise Exception('a exception')
    # except Exception as e:
    #     logger2.error("test catch exception", exc_info=True)
    # logger2.debug('test debug1')
    # logger2.debug('test debug2')
