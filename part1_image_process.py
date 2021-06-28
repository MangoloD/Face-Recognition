import random
import time
import numpy as np
import pandas as pd
from PIL import Image
from multiprocessing.pool import Pool
from tool import utils
from data_enhance import rand_method


def get_picture():
    info = pd.read_csv("Data/Document/data.csv")
    return info


"""数据追加记得改成a+"""


def division_picture(info, main_path, document_path, picture_path, picture_size, picture_num, picture_type, number):
    doc_positive = open(f"{picture_path}/{document_path[0]}", "w+")
    doc_part = open(f"{picture_path}/{document_path[1]}", "w+")
    doc_negative = open(f"{picture_path}/{document_path[2]}", "w+")

    for j in range(len(info)):
        filename, x, y, w, h, l_eye_x, l_eye_y, r_eye_x, r_eye_y, \
        nose_x, nose_y, l_mouth_x, l_mouth_y, r_mouth_x, r_mouth_y = info.loc[number + j]

        if w / h < 0.7 or h / w < 0.7:  # 7.654289484024048
            continue
        boxes = [[x, y, x + w, y + h]]

        positive_num = picture_num[0]
        part_num = picture_num[1]
        negative_num = picture_num[2]
        file_path = f"{main_path}/{filename}"
        image = Image.open(file_path)
        img_w, img_h = image.size
        # 人脸中心点位置
        cx = x + w / 2
        cy = y + h / 2
        side_len = random.choice([w, h])

        s = time.time()
        while True:
            # _cx = _cy = _side_len = 0
            # 正样本生成
            if positive_num > 0:
                _side_len = side_len + side_len * random.uniform(-0.2, 0.2) + 1
                _cx = cx + cx * random.uniform(-0.2, 0.2) + 1
                _cy = cy + cy * random.uniform(-0.2, 0.2) + 1

            # 部分样本生成
            elif part_num > 0:
                _side_len = side_len + side_len * random.uniform(-1, 1) + 1
                _cx = cx + cx * random.uniform(-1, 1) + 1
                _cy = cy + cy * random.uniform(-1, 1) + 1

            x_ = _cx - _side_len / 2
            y_ = _cy - _side_len / 2

            # 负样本生成
            n_w = random.randint(int(min(img_w, img_h) * 0.35),
                                 int(min(img_w, img_h) * 0.40))
            n_x = random.randint(0, img_w - n_w)
            n_y = random.randint(0, img_h - n_w)

            new_box = [x_, y_, x_ + _side_len, y_ + _side_len]
            ratio = utils.iou(new_box, np.array(boxes))[0]

            n_new_box = [n_x, n_y, n_x + n_w, n_y + n_w]
            n_ratio = utils.iou(n_new_box, np.array(boxes))[0]

            offset_x1 = (x - x_) / _side_len
            offset_y1 = (y - y_) / _side_len
            offset_x2 = ((x + w) - (x_ + _side_len)) / _side_len
            offset_y2 = ((y + h) - (y_ + _side_len)) / _side_len

            offset_px1 = (l_eye_x - x_) / _side_len
            offset_py1 = (l_eye_y - y_) / _side_len
            offset_px2 = (r_eye_x - x_) / _side_len
            offset_py2 = (r_eye_y - y_) / _side_len
            offset_px3 = (nose_x - x_) / _side_len
            offset_py3 = (nose_y - y_) / _side_len
            offset_px4 = (l_mouth_x - x_) / _side_len
            offset_py4 = (l_mouth_y - y_) / _side_len
            offset_px5 = (r_mouth_x - x_) / _side_len
            offset_py5 = (r_mouth_y - y_) / _side_len

            name = filename.split('.')[0]
            img = image.crop((x_, y_, x_ + _side_len, y_ + _side_len))
            img = img.resize((picture_size, picture_size))
            if ratio > 0.7 and positive_num > 0:
                if positive_num == 1:
                    number_ = random.randint(2, 3)
                    img = rand_method(img, number_, picture_size)
                new_filename = f"{name}_{picture_size}_{positive_num}.jpg"
                img.save(f"{picture_path}/{picture_type[0]}/{new_filename}")
                doc_positive.write(f"{picture_type[0]}/{new_filename} 1 "
                                   f"{offset_x1} {offset_y1} {offset_x2} {offset_y2} "
                                   f"{offset_px1} {offset_py1} {offset_px2} {offset_py2} "
                                   f"{offset_px3} {offset_py3} "
                                   f"{offset_px4} {offset_py4} {offset_px5} {offset_py5}\n")
                positive_num -= 1
            if 0.3 < ratio < 0.6 and part_num > 0:
                if part_num == 1:
                    number_ = random.randint(2, 3)
                    img = rand_method(img, number_, picture_size)
                new_filename = f"{name}_{picture_size}_{part_num}.jpg"
                img.save(f"{picture_path}/{picture_type[1]}/{new_filename}")
                doc_part.write(f"{picture_type[1]}/{new_filename} 2 "
                               f"{offset_x1} {offset_y1} {offset_x2} {offset_y2} "
                               f"{offset_px1} {offset_py1} {offset_px2} {offset_py2} "
                               f"{offset_px3} {offset_py3} "
                               f"{offset_px4} {offset_py4} {offset_px5} {offset_py5}\n")
                part_num -= 1
            if n_ratio < 0.1 and negative_num > 0:
                img = image.crop((n_x, n_y, n_x + n_w, n_y + n_w))
                img = img.resize((picture_size, picture_size))
                new_filename = f"{name}_{picture_size}_{negative_num}.jpg"
                img.save(f"{picture_path}/{picture_type[2]}/{new_filename}")
                doc_negative.write(f"{picture_type[2]}/{new_filename} 0 "
                                   f"{0} {0} {0} {0} "
                                   f"{0} {0} {0} {0} "
                                   f"{0} {0} "
                                   f"{0} {0} {0} {0}\n")
                negative_num -= 1
            if positive_num == 0 and part_num == 0 and negative_num == 0 or (time.time() - s) > 1:
                break
        print(filename)
    doc_positive.close()
    doc_part.close()
    doc_negative.close()


if __name__ == '__main__':
    pic_main_path = "G:/FeigeDownload/img_celeba/img_celeba"
    root_path = ["G:/MTCNN_Data/Dataset_1/Train_Image",
                 "G:/MTCNN_Data/Dataset_1/Validate_Image"]
    # root_path = ["G:/Dataset/Train_Image"]
    pic_size_l = [12, 24, 48]
    pic_num = [3, 3, 9]
    pic_type = ["positive", "part", "negative"]
    doc_name = [["train_positive.txt", "train_part.txt", "train_negative.txt"],
                ["validate_positive.txt", "validate_part.txt", "validate_negative.txt"],
                ["test_positive.txt", "test_part.txt", "test_negative.txt"]]

    info_list = get_picture()[:30000]
    length = len(info_list)
    tra_num = int(length * 0.8)
    val_num = int(length * 0.05)
    tes_num = length - tra_num - val_num
    every_info = (info_list[0:tra_num],
                  info_list[tra_num:tra_num + val_num],
                  info_list[tra_num + val_num:])

    num = [0, len(every_info[0]), len(every_info[0]) + len(every_info[1])]

    pool = Pool(9)
    start = time.perf_counter()
    for i, r_path in enumerate(root_path):
        doc_path = doc_name[i]
        for pic_size in pic_size_l:
            pic_path = f"{r_path}/{str(pic_size)}"
            # division_picture(every_info[i], pic_main_path, doc_path,
            #                  pic_path, pic_size, pic_num, pic_type, num[i])
            pool.apply_async(division_picture,
                             args=(every_info[i], pic_main_path, doc_path,
                                   pic_path, pic_size, pic_num, pic_type, num[i]))
    pool.close()
    pool.join()
    end = time.perf_counter()
    print()
    print(end - start)
