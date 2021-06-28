import os
import time
import cv2
import torch
import numpy as np
import pandas as pd
import pickle
import torch.nn.functional as f
import torchvision.transforms as transforms
from PIL import Image, ImageFont, ImageDraw

from face_net import FaceNet
from part6_detect import Detector


def compare(face1, face2):
    face1_normal = f.normalize(face1)
    face2_normal = f.normalize(face2)

    cos_a = torch.matmul(face1_normal, face2_normal.T)

    return cos_a


def feature_save(transform, net, file_path, load_path, save_path):
    doc = open(save_path, "ab+")
    net.load_state_dict(torch.load(load_path))  # path指的是face.py文件中保存的参数的路径
    net.eval()
    persons_faces = []  # 建立人脸库的列表
    for person in os.listdir(file_path):  # 遍历每一个人脸文件夹
        person_face = []  # 用来同一个人的不同的人脸特征（一个人获取的可能不止一张照片）
        # persons_name.append(person)  # 存放人的名字
        for face in os.listdir(os.path.join(file_path, person)):  # 人脸照片转换为特征
            person_picture = transform(Image.open(os.path.join(file_path, person, face))).cuda()
            person_feature = net.encode(person_picture[None, ...])  # 获取编码后的每一个人的脸部特征
            feature = person_feature.detach().cpu()  # 将脸部特征转到CPU上，节省GPU的计算量
            person_face.append(feature)  # 将同一个人脸的不同人脸特征存放到同一个列表中
        # persons_faces[person] = person_face  #
        persons_faces.append([person, torch.cat(person_face, dim=0)])  # 将不同人的名字、脸部特征存放到同一个列表中
    pickle.dump(persons_faces, doc)  # 按照列表形式存入文件


def identity(load_path, picture1_path, picture2_path):
    tf = transforms.Compose([
        transforms.Resize(112),
        transforms.ToTensor(),
    ])
    net = FaceNet().cuda()
    net.load_state_dict(torch.load(load_path))
    net.eval()

    person1 = tf(Image.open(picture1_path)).cuda()
    person1_feature = net(person1[None, ...])[0]

    person2 = tf(Image.open(picture2_path)).cuda()
    person2_feature = net(person2[None, ...])[0]

    siam = compare(person1_feature, person2_feature)
    print(siam.item())


def video_call(load_path):
    tf = transforms.Compose([
        # transforms.Resize(112),
        transforms.ToTensor(),
    ])

    net = FaceNet().cuda()
    net.load_state_dict(torch.load(load_path))
    net.eval()

    # persons_name = []  # 建立不同人物的名字（文件夹名）
    # persons_faces = []  # 建立人脸库的列表
    # file_path = r"G:\DL_Data\Star"  # 人脸存放路径
    # for person in os.listdir(file_path):  # 遍历每一个人脸文件夹
    #     person_face = []  # 用来同一个人的不同的人脸特征（一个人获取的可能不止一张照片）
    #     persons_name.append(person)  # 存放人的名字
    #
    #     for face in os.listdir(os.path.join(file_path, person)):  # 人脸照片转换为特征
    #         person_picture = tf(Image.open(os.path.join(file_path, person, face))).cuda()
    #         person_feature = net.encode(person_picture[None, ...])  # 获取编码后的每一个人的脸部特征
    #         feature = person_feature.detach().cpu()  # 将脸部特征转到CPU上，节省GPU的计算量
    #         person_face.append(feature)  # 将同一个人脸的不同人脸特征存放到同一个列表中
    #     persons_faces.append(person_face)  # 将不同人的脸部特征存放到同一个列表中
    font_path = r"C:\Windows\Fonts\simhei.ttf"  # 设置字体的路径
    font1 = ImageFont.truetype(font_path, 19, encoding="utf-8")  # 设置字体的格式

    cap = cv2.VideoCapture(0)  # 读取视频流
    cap.set(4, 720)  # 设置读取的视频的宽和高
    cap.set(3, 480)
    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))  # 帧率

    while True:
        t1 = time.time()  # 计时器
        ret, im = cap.read()
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):
            break
        elif ret is False:
            break

        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # 特征对比之前先转换通道，因为OpenCV读取的通道格式是BGR
        im = Image.fromarray(np.uint8(im))  # 转为PIL类型
        with torch.no_grad():  # 此处不用梯度
            detector1 = Detector("Data/Params_/P/p_net_49.pth",
                                 "Data/Params_2/R/r_net_26.pth",
                                 "Data/Params_2/O/o_net_36.pth")  # MTCNN人脸检测器实例化
            boxes = detector1.detect(im)  # 读取到的视频流传入检测器中

            for box in boxes:  # 检测到的人脸的四个坐标值
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])
                w = x2 - x1
                h = y2 - y1

                if type(im) is np.ndarray:  # 读取到的图像有时会报类型错误，避免报错
                    im = Image.fromarray(np.uint8(im))
                    cropped = im.crop((x1, y1, x2, y2))  # 使用PIL对图像进行裁剪
                    draw = ImageDraw.Draw(im)  # 在图像上画出目标框
                    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0))
                else:
                    cropped = im.crop((x1, y1, x2, y2))  # 使用PIL对图像进行裁剪
                    draw = ImageDraw.Draw(im)  # 在图像上画目标框
                    draw.rectangle([x1, y1, x2 - 0.1 * w, y2 - 0.2 * h], outline=(255, 0, 0))

                person1 = tf(cropped).cuda()  # 将MTCNN裁剪出来的图片归一化并且传入cuda
                person1_feature = net.encode(person1[None, ...])  # 获取到处理后的视频人脸的特征

                persons_similarity = []  # 建立一个空列表用来存放不同人的脸部特征与视频人脸特征的余弦相似度

                doc = open(save_path_, "rb")
                person = pickle.load(doc)  # 读取列表形式
                for personal_face in person:
                    personal_name = personal_face[0]
                    personal_features = personal_face[1].cuda()
                    siam = compare(person1_feature, personal_features)
                    sia = max(siam[0]).item()
                    persons_similarity.append([personal_name, sia])
                data = pd.DataFrame(persons_similarity)
                data = data.sort_values(by=1, ascending=False)
                obj_name = data.iloc[0][0]

                # for personal_face in person:  # 遍历不同人的不同人脸特征
                #     similarity = []  # 建立一个列表存放同一个人的不同脸部特征与视频人脸的相似性
                #     for face in personal_face:  # 遍历同一个人的不同的人脸特征
                #         person2_feature = face.cuda()  # 将人脸特征传入cuda
                #         siam = compare(person1_feature, person2_feature)  # 与一个人的一张人脸特征做比较
                #
                #         im = np.array(im).astype("uint8")  # 转为uint8类型
                #         similarity.append(siam.item())  # 同一个人的所有脸特征与当前视频人脸的余弦相似度
                #     personal_max = max(similarity)  # 选择最大的余弦相似度
                #     persons_similarity.append(personal_max)  # 取相似性最高的（同一个人的不同人脸特征中与视频人脸作对比，取出最大的余弦相似度）
                # sia = max(persons_similarity)  # 将不同人的脸部特征与视频人脸对比出的最大的相似度存放在列表中（此时是本地库有几个人就有几个相似度）
                # obj_name = person.get(sia)
                # obj_name = persons_name[sia]
                # idx = persons_similarity.index(sia)  # 确定最大的相似的人脸的索引位置
                # obj_name = persons_name[idx]  # 索引对应目标人物的名字

                im = np.array(im).astype("uint8")
                if sia > 0.2:  # 当相似度大于某个值时才显示name和相似度
                    cv2.putText(im, str(float("%.2f" % sia)), (x1, y1),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 0), 2)  # 在图像上写最大相似度
                    cv2.putText(im, str(obj_name), (x1, int((y1 + 15))),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)  # 写上最大相似度的标签（最像谁）
                else:
                    im = Image.fromarray(im)
                    draw = ImageDraw.Draw(im)
                    draw.text((x1, int(y1 + 15)), "Who?", (255, 0, 0), font=font1)
            im = cv2.cvtColor(np.uint8(im), cv2.COLOR_RGB2BGR)  # 展示视频流前转换通道
            cv2.imshow("", np.uint8(im))
            t2 = time.time()
            print(t2 - t1)


if __name__ == '__main__':
    load_params = "Data/30.pth"
    pic1_path = "G:/DL_Data/star_face/val/116/116-91.jpg"

    pic2_path = "G:/DL_Data/star_face/val/116/116-94.jpg"
    # pic1_path = "../Document/test_img/pic0.jpg"
    # pic2_path = "../Document/test_img/pic146.jpg"
    # identity(load_params, pic1_path, pic2_path)

    tf_ = transforms.Compose([
        # transforms.Resize(112),
        transforms.ToTensor(),
    ])

    net_ = FaceNet().cuda()
    file_path_ = "G:/DL_Data/Star"
    save_path_ = "Data/face.data"
    # feature_save(tf_, net_, file_path_, load_params, save_path_)
    video_call(load_params)
