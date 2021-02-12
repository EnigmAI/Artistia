from PIL import Image
import os
def pixelate(pixel_size = 4):
    path = 'static/uploads'
    img_path = os.path.join(path, "source3.png")
    image = Image.open(img_path)
    print(image.size)
    image = image.resize(
        (image.size[0] // pixel_size, image.size[1] // pixel_size),
        Image.NEAREST
    )
    image = image.resize(
        (image.size[0] * pixel_size, image.size[1] * pixel_size),
        Image.NEAREST
    )
    fname = "static/results/result_pixel.png"
    image.save(fname)

