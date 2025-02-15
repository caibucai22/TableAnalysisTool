from service.eval.BingoClsService import BingoClsService, BingoClsServiceV2


def test_binary_cls():
    img_path = (
        "E:/my-github-repos/01-my_repos/base_output/cache/locate_table_3/cell_5_2.jpg"
    )
    model = BingoClsService()
    bingo_ret = model.binary_cls([img_path])
    print(bingo_ret)


def test_bingo_det():
    img_path = "E:/my-github-repos/01-my_repos/base_output/cache/locate_table_2/1.jpg"
    model = BingoClsServiceV2()
    bingo_ret = model.binary_cls([img_path])
    print(bingo_ret)


if __name__ == "__main__":
    test_binary_cls()
    test_bingo_det()
    # pass
