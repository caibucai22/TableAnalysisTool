from models.TableTransformer import TableTransformer
from PIL import Image


def test_table_transformer():
    model = TableTransformer()
    img_path = "./test_images/table0.jpg"
    img = Image.open(img_path).convert("RGB")
    results = model.infer(img)
    for k, v in results.items():
        print(k)


if __name__ == "__main__":
    test_table_transformer()
