from PIL import Image
from pathlib import Path


def png_to_ico(input_path, output_path=None):
    input_path = Path(input_path)

    if output_path is None:
        output_path = input_path.with_suffix(".ico")

    img = Image.open(input_path)
    img.thumbnail((512, 512))

    size = (512, 512)
    img_square = Image.new("RGBA", size, (0, 0, 0, 0))

    w, h = img.size
    offset = ((size[0] - w) // 2, (size[1] - h) // 2)
    img_square.paste(img, offset)

    img_square.save(
        output_path,
        format="ICO",
        sizes=[(16,16),(32,32),(48,48),(64,64),(128,128),(256,256),(512,512)]
    )

    print(f"Icon created: {output_path}")


if __name__ == "__main__":
    png_to_ico(r"src\cmj_framework\gui\assets\icons\cmj_logo.png")