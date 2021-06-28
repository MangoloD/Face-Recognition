import random
import time
from PIL import Image
import pandas as pd

from del_doc_data import del_same


def get_picture():
    return pd.read_table("Data/Document/list_bbox_celeba.txt", sep="\s+", header=1, low_memory=False)


def division_picture(info, main_path, document_path, picture_path, picture_size, picture_num, number):
    doc_positive = open(f"{main_path}/{document_path[0]}", "a+")
    doc_part = open(f"{main_path}/{document_path[1]}", "a+")
    doc_negative = open(f"{main_path}/{document_path[2]}", "a+")
    # doc_error = open(error_path, "a+")

    for j in range(len(info)):
        filename, x, y, w, h = info.loc[number + j]
        print(filename, x, y, w, h)
        positive_num = picture_num[0]
        part_num = picture_num[1]
        negative_num = picture_num[2]

        file_path = f"{pic_main_path}/{filename}"
        image = Image.open(file_path)
        img_w, img_h = image.size

        if (w / h) < 0.7 or (h / w) < 0.7:
            # print(filename)
            # text = f"{filename} {x} {y} {w} {h}\n"
            # del_same(doc_error, text)
            continue
        exit()
        while True:
            w_ = h_ = int(min(w, h) * random.uniform(0.8, 1.2))

            x_ = random.randint(int(x - w * 0.1), int(x + w * 0.1))
            y_ = random.randint(int(y - h * 0.1), int(y + h * 0.1))

            n_w = random.randint(int(min(img_w, img_h) * 0.35),
                                 int(min(img_w, img_h) * 0.45))
            n_x = random.randint(0, img_w - n_w)
            n_y = random.randint(0, img_h - n_w)

            if x_ + w_ > img_w:
                x_ = img_w - w_ - 10
            if x_ < 0:
                x_ = 10
            if y_ + h_ > img_h:
                y_ = img_w - h_ - 10
            if y_ < 0:
                y_ = 10
            x1, y1 = max(x, x_), max(y, y_)
            x2, y2 = min(x + w, x_ + w_), min(y + h, y_ + h_)

            if x2 - x1 > 0 and y2 - y1 > 0:
                s_intersection = (x2 - x1) * (y2 - y1)
            else:
                s_intersection = 0
            s_union = w * h + w_ * w_ - s_intersection
            ratio = s_intersection / s_union

            n_x1, n_y1 = max(x, n_x), max(y, n_y)
            n_x2, n_y2 = min(x + w, n_x + n_w), min(y + h, n_y + n_w)

            if n_x2 - n_x1 > 0 and n_y2 - n_y1 > 0:
                n_intersection = (n_x2 - n_x1) * (n_y2 - n_y1)
            else:
                n_intersection = 0
            n_union = w * h + w_ * w_ - n_intersection
            n_ratio = n_intersection / n_union

            offset_x1 = (x - x_) / w_
            offset_y1 = (y - y_) / h_
            offset_x2 = ((x + w) - (x_ + w_)) / w_
            offset_y2 = ((y + h) - (y_ + h_)) / h_

            offset_n_x1 = (x - n_x) / n_w
            offset_n_y1 = (y - n_y) / n_w
            offset_n_x2 = ((x + w) - (n_x + n_w)) / n_w
            offset_n_y2 = ((y + h) - (n_y + n_w)) / n_w

            # draw = ImageDraw.Draw(image)
            # draw.rectangle(((x, y), (x + w, y + h)), outline="red", width=3)
            # draw.rectangle(((x_, y_), (x_ + w_, y_ + h_)), outline="orange", width=3)
            # draw.rectangle(((n_x, n_y), (n_x + n_w,
            #                              n_y + n_w)), outline="green", width=3)
            # plt.xticks([])
            # plt.yticks([])
            # plt.title(filename)
            # plt.imshow(image)
            # print(ratio, n_ratio)
            # plt.pause(0.1)
            # plt.clf()

            name = filename.split('.')[0]
            if ratio > 0.7 and positive_num > 0:
                img = image.crop((x_, y_, x_ + w_, y_ + h_))
                img = img.resize((picture_size, picture_size))
                new_filename = f"{name}_{picture_size}_{positive_num}.jpg"
                img.save(f"{picture_path}/positive/{new_filename}")
                doc_positive.write(f"{new_filename} 1 {offset_x1} {offset_y1} {offset_x2} {offset_y2}\n")
                positive_num -= 1
            if 0.3 < ratio < 0.6 and part_num > 0:
                img = image.crop((x_, y_, x_ + w_, y_ + h_))
                img = img.resize((picture_size, picture_size))
                new_filename = f"{name}_{picture_size}_{part_num}.jpg"
                img.save(f"{picture_path}/part/{new_filename}")
                doc_part.write(f"{new_filename} 2 {offset_x1} {offset_y1} {offset_x2} {offset_y2}\n")
                part_num -= 1
            if n_ratio < 0.1 and negative_num > 0:
                img = image.crop((n_x, n_y, n_x + n_w, n_y + n_w))
                img = img.resize((picture_size, picture_size))
                new_filename = f"{name}_{picture_size}_{negative_num}.jpg"
                img.save(f"{picture_path}/negative/{new_filename}")
                doc_negative.write(f"{new_filename} 0 {offset_n_x1} {offset_n_y1} {offset_n_x2} {offset_n_y2}\n")
                negative_num -= 1
            if positive_num == 0 and part_num == 0 and negative_num == 0:
                break
    doc_positive.close()
    doc_part.close()
    doc_negative.close()
    # doc_error.close()


if __name__ == '__main__':
    pic_main_path = "G:/FeigeDownload/img_celeba/img_celeba"  # "Resource"
    root_path = ["G:/Data/Image/Train_Image", "G:/Data/Image/Validate_Image", "G:/Data/Image/Test_Image"]
    pic_size_l = [12, 24, 48]
    pic_num = [3, 3, 9]
    pic_type = ["positive", "part", "negative"]
    doc_name = [["train_positive.txt", "train_part.txt", "train_negative.txt"],
                ["validate_positive.txt", "validate_part.txt", "validate_negative.txt"],
                ["test_positive.txt", "test_part.txt", "test_negative.txt"]]
    error_path = "G:/Data/Image/error_data.txt"

    info_list = get_picture()[:160]
    length = len(info_list)
    tra_num = int(length * 0.7)
    val_num = int(length * 0.15)
    tes_num = length - tra_num - val_num

    every_info = (info_list[0:tra_num],
                  info_list[tra_num:tra_num + val_num],
                  info_list[tra_num + val_num:])

    num = [0, len(every_info[0]), len(every_info[0]) + len(every_info[1])]

    start = time.time()
    for i, r_path in enumerate(root_path):
        doc_path = doc_name[i]
        for pic_size in pic_size_l:
            pic_path = f"{r_path}/{str(pic_size)}"
            division_picture(every_info[i], r_path, doc_path, pic_path, pic_size, pic_num, num[i])
        print(f"{r_path.split('/')[3]} is over...")
    end = time.time()
    print()
    print(end - start)
