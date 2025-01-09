import logging
from Logger import get_logger

logger = get_logger("TableEntity", log_level=logging.DEBUG)


class TableEntityError(Exception):
    pass


class Table:

    def __init__(
        self,
        row_bbox_list,
        col_bbox_list,
        cell_bbox_list,
        type="wired",
        skip_table_head=True,
    ) -> None:
        if len(row_bbox_list) * len(col_bbox_list) != len(cell_bbox_list):
            raise TableEntityError("rows,cols don't match cells")
        self.type = type  # wired no-wired
        self.n_row = len(row_bbox_list)
        self.n_col = len(col_bbox_list)
        self.n_cell = len(cell_bbox_list)
        self.row_bbox_list = row_bbox_list
        self.col_bbox_list = col_bbox_list
        self.cell_bbox_list = cell_bbox_list
        self.skip_table_head = skip_table_head  # for filter first row cell
        self.cell_ocr_list = []

    def get_row_list(self):
        return self.row_list

    def get_col_list(self):
        return self.col_list

    def get_cell_list(self):
        return self.cell_list

    def set_ocr_rec(self, ocr_res_list):
        self.cell_ocr_list = ocr_res_list

    def get_cell_ocr_list(self):
        if len(self.cell_ocr_list) == 0:
            logger.warning("cell ocr res list is empty")
        return self.cell_ocr_list

    def __repr__(self):
        """
        Returns a string representation of the Table object.
        """
        return f"Table(type='{self.table_type}', rows={self.n_row}, cols={self.n_col}, cells={self.n_cell})"

    def is_matched(self, gt_n_row, gt_n_col):
        if self.n_col != gt_n_col or self.n_row != gt_n_row:
            logger.warning(
                "parsed table structure don't match groundtruth table structure"
            )
            return False
        return True
