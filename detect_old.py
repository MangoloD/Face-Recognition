import torch
from PIL import Image
from PIL import ImageDraw
import numpy as np
from tool import utils
import part2_nets
from torchvision import transforms
import time


class Detector:

    def __init__(self, pnet_param="Data/Params_/P/p_net_12.pth",
                 rnet_param="Data/Params/R/r_net_13.pth",
                 onet_param="Data/Params/O/o_net_2.pth",
                 isCuda=False):

        self.isCuda = isCuda

        self.pnet = part2_nets.PNet()
        self.rnet = part2_nets.RNet()
        self.onet = part2_nets.ONet()

        if self.isCuda:
            self.pnet.cuda()
            self.rnet.cuda()
            self.onet.cuda()

        self.pnet.load_state_dict(torch.load(pnet_param, map_location='cpu'))
        self.rnet.load_state_dict(torch.load(rnet_param, map_location='cpu'))
        self.onet.load_state_dict(torch.load(onet_param, map_location='cpu'))

        self.pnet.eval()
        self.rnet.eval()
        self.onet.eval()

        self.__image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def detect(self, image):

        start_time = time.time()
        pnet_boxes = self.__pnet_detect(image)
        # print(pnet_boxes)
        if pnet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        t_pnet = end_time - start_time

        start_time = time.time()
        rnet_boxes = self.__rnet_detect(image, pnet_boxes)
        # print( rnet_boxes)
        if rnet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        t_rnet = end_time - start_time

        start_time = time.time()
        onet_boxes = self.__onet_detect(image, rnet_boxes)
        if onet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        t_onet = end_time - start_time

        t_sum = t_pnet + t_rnet + t_onet

        print("total:{0} pnet:{1} rnet:{2} onet:{3}".format(t_sum, t_pnet, t_rnet, t_onet))

        return pnet_boxes

    def __pnet_detect(self, img):

        boxes = []
        w, h = img.size
        min_side_len = min(w, h)

        scale = 1

        while min_side_len >= 12:
            img_data = self.__image_transform(img)
            if self.isCuda:
                img_data = img_data.cuda()
            img_data.unsqueeze_(0)

            # 由于P网络输出是全卷积，所以的到的cls和offset都是四维的NCHW
            _cls, _offest = self.pnet(img_data)

            # NCHW→HW，取每张图的第一个值C的高宽,因为输入的图像大于12*12，所以特征图是大于1*1的，所以得到的特征图就是H*W的，
            # 而每个H,W对于的特征点反算回原图都有对于每个区域的置信度，所以HW是置信度的集合
            # NCHW→CHW，计算每张图上的坐标偏移率，也就是输出的4个通道上（x1,y1,x2,y2）偏移率的集合
            cls, offest = _cls[0][0].cpu().data, _offest[0].cpu().data
            # print(cls.shape)#[H,W]
            # print(offest.shape)#[C,H,W]
            # 得到置信度大于阈值的每组(h,w)索引值，就可以返回算同一个索引的置信度和偏移率
            idxs = torch.nonzero(torch.gt(cls, 0.6))
            # 遍历达标的索引库，得到每组(h,w)索引
            for idx in idxs:
                # print(idx)
                # 传入每组索引(h,w)，偏移量，索引对应的置信度，比例
                boxes.append(self.__box(idx, offest, cls, scale))
            scale *= 0.709
            _w = int(w * scale)
            _h = int(h * scale)

            img = img.resize((_w, _h))
            # print(min_side_len)
            min_side_len = np.minimum(_w, _h)
            # print(min_side_len)
            # boxss = utils.nms(np.array(boxes), 0.3)
        # return boxss
        return utils.nms(np.array(boxes), 0.3)

    # 将偏移回归量还原到原图上去
    def __box(self, start_index, offset, cls, scale, stride=2, side_len=12):
        # print((start_index[1] * stride)/scale)
        # print(int(start_index[1] * stride+0)/scale)
        # 计算出建议框(训练样本)在原图上的坐标值
        _x1 = int(start_index[1] * stride) / scale  # 宽，W，x
        _y1 = int(start_index[0] * stride) / scale  # 高，H,y
        _x2 = int(start_index[1] * stride + side_len) / scale
        _y2 = int(start_index[0] * stride + side_len) / scale

        ow = _x2 - _x1  # 12
        oh = _y2 - _y1  # 12
        # print(ow,oh)

        # 计算出实际框在原图上的坐标值
        # _offset取所有通道上高和宽的索引位置对应的偏移值，就得到了一组四个值的坐标偏移率
        # print(offset.shape)
        _offset = offset[:, start_index[0], start_index[1]]
        # print(_offset.shape)
        # print(_offset)
        # x1 = _x1 + 12 * _offset[0]
        # y1 = _y1 + 12 * _offset[1]
        # x2 = _x2 + 12 * _offset[2]
        # y2 = _y2 + 12 * _offset[3]
        x1 = _x1 + ow * _offset[0]
        y1 = _y1 + oh * _offset[1]
        x2 = _x2 + ow * _offset[2]
        y2 = _y2 + oh * _offset[3]
        # print([x1, y1, x2, y2, cls])
        cls = cls[start_index[0], start_index[1]]
        return [x1, y1, x2, y2, cls]

    def __rnet_detect(self, image, pnet_boxes):

        _img_dataset = []
        _pnet_boxes = utils.convert_to_square(pnet_boxes)
        for _box in _pnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((24, 24))
            img_data = self.__image_transform(img)
            _img_dataset.append(img_data)

        img_dataset = torch.stack(_img_dataset)
        if self.isCuda:
            img_dataset = img_dataset.cuda()

        _cls, _offset = self.rnet(img_dataset)

        _cls = _cls.cpu().data.numpy()
        # print(cls)
        # print(cls.shape) #[N,1]
        offset = _offset.cpu().data.numpy()
        # print(offset.shape) #[N,4]

        boxes = []
        # 选取置信度达标的索引
        idxs, _ = np.where(_cls > 0.6)
        # print(idxs)
        # print(idxs.shape) #[N,]
        for idx in idxs:
            # 使用R网络的置信度来筛选P网络的截图
            _box = _pnet_boxes[idx]
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            ow = _x2 - _x1
            oh = _y2 - _y1

            x1 = _x1 + ow * offset[idx][0]
            y1 = _y1 + oh * offset[idx][1]
            x2 = _x2 + ow * offset[idx][2]
            y2 = _y2 + oh * offset[idx][3]
            cls = _cls[idx][0]

            boxes.append([x1, y1, x2, y2, cls])

        return utils.nms(np.array(boxes), 0.3)

    def __onet_detect(self, image, rnet_boxes):

        _img_dataset = []
        _rnet_boxes = utils.convert_to_square(rnet_boxes)
        for _box in _rnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((48, 48))
            img_data = self.__image_transform(img)
            _img_dataset.append(img_data)

        img_dataset = torch.stack(_img_dataset)
        if self.isCuda:
            img_dataset = img_dataset.cuda()

        _cls, _offset = self.onet(img_dataset)

        _cls = _cls.cpu().data.numpy()
        offset = _offset.cpu().data.numpy()

        boxes = []
        idxs, _ = np.where(_cls > 0.95)
        for idx in idxs:
            _box = _rnet_boxes[idx]
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            ow = _x2 - _x1
            oh = _y2 - _y1

            x1 = _x1 + ow * offset[idx][0]
            y1 = _y1 + oh * offset[idx][1]
            x2 = _x2 + ow * offset[idx][2]
            y2 = _y2 + oh * offset[idx][3]
            cls = _cls[idx][0]

            boxes.append([x1, y1, x2, y2, cls])

        return utils.nms(np.array(boxes), 0.3, is_min=True)


if __name__ == '__main__':
    x = time.time()
    with torch.no_grad() as grad:
        image_file = "Data/Test_Image/MTCNN1/10.jpg"
        detector = Detector()

        with Image.open(image_file) as im:
            boxes = detector.detect(im)
            print(boxes.shape)
            imDraw = ImageDraw.Draw(im)
            for box in boxes:
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])

                # print(box[4])
                imDraw.rectangle((x1, y1, x2, y2), outline='red', width=3)
            y = time.time()
            print(y - x)
            im.show()
