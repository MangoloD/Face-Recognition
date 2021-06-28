import os
import shutil


def del_data(filepath):
    f = open(filepath, 'w')
    # 清空文件
    f.truncate()
    f.close()


def del_pic(filepath):
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    else:
        shutil.rmtree(filepath)
        os.makedirs(filepath)


def del_same(file, line):
    file.seek(0)
    lines = file.readlines()
    if line in lines:
        pass
    else:
        file.write(f"{line}\n")


if __name__ == '__main__':
    root_path = ["G:/MTCNN_Data/Dataset_1/Train_Image",
                 "G:/MTCNN_Data/Dataset_1/Validate_Image",
                 "G:/MTCNN_Data/Dataset_1/Test_Image"]
    # root_path = ["Data/Image/Train_Image", "Data/Image/Validate_Image", "Data/Image/Test_Image"]
    pic_size_l = [12, 24, 48]
    pic_type = ["positive", "part", "negative"]
    doc_name = [["train_positive.txt", "train_part.txt", "train_negative.txt"],
                ["validate_positive.txt", "validate_part.txt", "validate_negative.txt"],
                ["test_positive.txt", "test_part.txt", "test_negative.txt"]]
    # error_path = "G:/Dataset/Image/error_data.txt"

    for i, r_path in enumerate(root_path):
        for p_size in pic_size_l:
            for j, p_type in enumerate(pic_type):
                pic_path = f"{r_path}/{p_size}/{p_type}"
                file_path = f"{r_path}/{p_size}/{doc_name[i][j]}"
                del_pic(pic_path)  # 清空文件夹图片
                del_data(file_path)  # 清空文档数据

    # del_data(error_path)  # 清空错误文件数据
