import random
import time
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def rand_method(image, num, face_size):
    method_list = ["bri_bool", "gau_bool", "noi_bool"]
    x = random.sample(method_list, num)

    if "bri_bool" in x:
        enh_bri = ImageEnhance.Brightness(image)
        brightness = random.uniform(0.4, 1.5)
        image = enh_bri.enhance(brightness)

    if "noi_bool" in x:
        width, height = image.size
        image = np.array(image)
        num = int(height * width * 0.06)  # 多少个像素点添加椒盐噪声
        for i in range(num):
            w = random.randint(0, width - 1)
            h = random.randint(0, height - 1)
            if random.randint(0, 1) == 0:
                image[h, w, 0] = random.randint(0, 255)
                image[h, w, 1] = random.randint(0, 255)
                image[h, w, 2] = random.randint(0, 255)
            else:
                image[h, w, 0] = random.randint(0, 255)
                image[h, w, 1] = random.randint(0, 255)
                image[h, w, 2] = random.randint(0, 255)
        image = Image.fromarray(image)

    if "gau_bool" in x:
        if face_size == 12:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1)))
        elif face_size == 24:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(1, 1.5)))
        elif face_size == 48:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.5, 2)))

    # if "col_bool" in x:
    #     enh_col = ImageEnhance.Color(image)
    #     color = random.uniform(0.8, 1.2)
    #     image = enh_col.enhance(color)
    return image


if __name__ == '__main__':
    picture_path = "G:/FeigeDownload/img_celeba/img_celeba/000003.jpg"
    image_ = Image.open(picture_path)
    s = time.time()
    number = random.randint(3, 3)
    # print(number)
    img = rand_method(image_, number, 48)
    img_ = rand_method(image_, 0, 24)
    print(time.time() - s)
    img.show()
    # img_.show()
