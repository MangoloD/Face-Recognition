import time
import torch
from torchvision import transforms
import numpy as np
from PIL import Image, ImageDraw
import part2_nets
from tool import utils


class Detector:
    def __init__(self, p_net_params, r_net_params, o_net_params, is_cuda=False):
        self.cuda = is_cuda
        self.p_net = part2_nets.PNet()
        self.r_net = part2_nets.RNet()
        self.o_net = part2_nets.ONet()

        if self.cuda:
            self.p_net.cuda()
            self.r_net.cuda()
            self.o_net.cuda()

        self.p_net.load_state_dict(torch.load(p_net_params, map_location="cpu"))
        self.r_net.load_state_dict(torch.load(r_net_params, map_location="cpu"))
        self.o_net.load_state_dict(torch.load(o_net_params, map_location="cpu"))

        self.p_net.eval()
        self.r_net.eval()
        self.o_net.eval()

        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # 将偏移回归量还原到原图上
    @staticmethod
    def box(start_index, loc, con, scale, stride=2, side_len=12):
        # 计算建议框（训练样本）在原图上的坐标值
        _x1 = start_index[:, 1] * stride / scale  # x
        _y1 = start_index[:, 0] * stride / scale  # y
        _x2 = (start_index[:, 1] * stride + side_len) / scale
        _y2 = (start_index[:, 0] * stride + side_len) / scale

        _w = _x2 - _x1
        _h = _y2 - _y1
        # print(_w, _h)

        # 计算出实际框在原图上的坐标值
        # _loc取所有通道上高和宽的索引值位置对应的偏移值，得到一组四个值的坐标偏移率
        _loc = loc[:, start_index[:, 0], start_index[:, 1]]
        # print(_loc, _loc.shape)
        x1_ = _x1 + _w * _loc[0, :]
        y1_ = _y1 + _h * _loc[1, :]
        x2_ = _x2 + _w * _loc[2, :]
        y2_ = _y2 + _h * _loc[3, :]

        con_ = con[start_index[:, 0], start_index[:, 1]]
        boxes = torch.stack([x1_, y1_, x2_, y2_, con_], dim=1)
        return np.array(boxes)

    def p_net_detect(self, image):
        _box = []
        w, h = image.size
        side = min(w, h)
        scale = 1

        while side >= 12:
            img_data = self.image_transform(image)
            if self.cuda:
                img_data = img_data.cuda()
            img_data = img_data.unsqueeze(0)

            # 由于P网络输出为全卷积，所以con、loc都是四维的NCHW
            _con, _loc = self.p_net(img_data)

            # NCHW→HW，取每张图的第一个值C的宽高，因为输入图像大于12*12，所以特征图大于1*1，所得特征图就为H*W
            # 每一个H、W对于特征点反算回原图都有对应每个区域的置信度，所以H、W是置信度的集合
            # NCHW→CHW，计算每张图的坐标偏移率，也就是输出的4个通道上(x1, y1, x2, y2)偏移率的集合
            con, loc = _con[0][0].cpu().data, _loc[0].cpu().data
            # print(con.shape, loc.shape)
            # 得到置信度大于阈值的每一组（h, w）的索引值，就可以反算同一索引的置信度和偏移率
            s_idx = torch.nonzero(torch.gt(con, 0.6))
            # 遍历达标索引值，得到每组（h, w）
            # for idx in s_idx:
            #     # print(idx)
            #     # 传入每组索引（h, w），偏移量，索引对应置信度，比例
            #     _box.append(self.box(idx, loc, con, scale))
            _box.extend(self.box(s_idx, loc, con, scale))

            scale *= 0.7
            _w = int(w * scale)
            _h = int(h * scale)
            image = image.resize((_w, _h))
            side = min(_w, _h)
        return utils.nms(np.array(_box), 0.3)

    def r_net_detect(self, image, p_net_boxes):
        _img_dataset = []
        _p_net_boxes = utils.convert_to_square(p_net_boxes)
        for _box in _p_net_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((24, 24))
            img_data = self.image_transform(img)
            _img_dataset.append(img_data)
        img_dataset = torch.stack(_img_dataset)

        if self.cuda:
            img_dataset = img_dataset.cuda()

        _con, _loc = self.r_net(img_dataset)
        _con, loc = _con.cpu().data.numpy(), _loc.cpu().data.numpy()
        # print(_con.shape, _loc.shape)
        # 选取置信度达标的索引
        s_idx, _ = np.where(_con > 0.8)

        _box = _p_net_boxes[s_idx]
        _x1, _y1, _x2, _y2 = _box[:, 0], _box[:, 1], _box[:, 2], _box[:, 3]
        _w = _x2 - _x1
        _h = _y2 - _y1
        x1 = _x1 + _w * loc[s_idx][:, 0]
        y1 = _y1 + _h * loc[s_idx][:, 1]
        x2 = _x2 + _w * loc[s_idx][:, 2]
        y2 = _y2 + _h * loc[s_idx][:, 3]
        cls_ = _con[s_idx][:, 0]
        boxes = np.stack([x1, y1, x2, y2, cls_], axis=1)

        return utils.nms(np.array(boxes), 0.3)

    def o_net_detect(self, image, r_net_boxes):
        _img_dataset = []
        _r_net_boxes = utils.convert_to_square(r_net_boxes)
        for _box in _r_net_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((48, 48))
            img_data = self.image_transform(img)
            _img_dataset.append(img_data)
        img_dataset = torch.stack(_img_dataset)
        if self.cuda:
            img_dataset = img_dataset.cuda()
        _con, _loc = self.o_net(img_dataset)

        _con, loc = _con.cpu().data.numpy(), _loc.cpu().data.numpy()

        s_idx, _ = np.where(_con > 0.99)

        _box = _r_net_boxes[s_idx]
        _x1, _y1, _x2, _y2 = _box[:, 0], _box[:, 1], _box[:, 2], _box[:, 3]
        _w = _x2 - _x1
        _h = _y2 - _y1
        x1 = _x1 + _w * loc[s_idx][:, 0]
        y1 = _y1 + _h * loc[s_idx][:, 1]
        x2 = _x2 + _w * loc[s_idx][:, 2]
        y2 = _y2 + _h * loc[s_idx][:, 3]
        cls_ = _con[s_idx][:, 0]

        boxes = np.stack([x1, y1, x2, y2, cls_], axis=1)

        return utils.nms(np.array(boxes), 0.3, is_min=True)

    def detect(self, image):
        start = time.time()
        p_net_boxes = self.p_net_detect(image)
        if p_net_boxes.shape[0] == 0:
            return np.array([])
        end = time.time()
        t_p_net = end - start

        start = time.time()
        r_net_boxes = self.r_net_detect(image, p_net_boxes)
        if r_net_boxes.shape[0] == 0:
            return np.array([])
        end = time.time()
        t_r_net = end - start

        start = time.time()
        o_net_boxes = self.o_net_detect(image, r_net_boxes)
        if o_net_boxes.shape[0] == 0:
            return np.array([])
        end = time.time()
        t_o_net = end - start

        t_sum = t_p_net + t_r_net + t_o_net

        print("total:{0} p_net:{1} r_net:{2} o_net:{3}".format(t_sum, t_p_net, t_r_net, t_o_net))

        return o_net_boxes


if __name__ == '__main__':
    x = time.time()
    with torch.no_grad() as grad:
        detector = Detector("Data/Params_/P/p_net_49.pth",
                            "Data/Params_2/R/r_net_26.pth",
                            "Data/Params_2/O/o_net_36.pth")
        for i in range(10):
            image_file = "Data/Test_Image/MTCNN2/{}.jpg".format(i + 1)
            with Image.open(image_file) as im:
                boxes_ = detector.detect(im)
                print(boxes_.shape)
                imDraw = ImageDraw.Draw(im)
                for box in boxes_:
                    _x11 = int(box[0])
                    _y11 = int(box[1])
                    _x22 = int(box[2])
                    _y22 = int(box[3])

                    # print(box[4])
                    # img = im.crop((_x11, _y11, _x22, _y22))
                    # img = img.resize((48, 48))
                    # img.show()
                    # exit()
                    imDraw.rectangle((_x11, _y11, _x22, _y22), outline='red', width=3)
                y = time.time()
                print(y - x)
                im.show()
