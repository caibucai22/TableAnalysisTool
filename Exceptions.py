class BaseException(Exception):
    def __init__(self, message) -> None:
        self.message = message

    # def __str__(self) -> str:
    #     return f"{self.message}"

class TableStructreError(BaseException):
    '''
    定位表格失败
    表格特征编码失败
    表格切分失败
    structure donot matched
    '''
    pass


class BingoError(BaseException):
    pass


class LineScoreError(BaseException):
    '''
    score list recover
    skipping
    '''
    pass


class OrderJudgeError(BaseException):
    pass

class OcrError(BaseException):
    '''
    '''
    pass



if __name__ == '__main__':
    # raise BingoError("no bingo")
    # raise BingoError("all bingos")
