import numpy as np


def iou(box, boxes, is_min=False):
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)

    inter = w * h
    if is_min:
        ratio = np.true_divide(inter, np.minimum(box_area, boxes_area))
    else:
        ratio = np.true_divide(inter, (box_area + boxes_area - inter))

    return ratio


def nms(boxes, thresh=0.3, is_min=False):
    if boxes.shape[0] == 0:
        return np.array([])

    confidence_boxes = boxes[(-boxes[:, 4]).argsort()]  # 置信度从大到小排序
    retain_boxes = []  # 需要保留的元素列表
    while confidence_boxes.shape[0] > 1:
        max_element = confidence_boxes[0]  # 获取最大值数据
        other_element_boxes = confidence_boxes[1:]  # 其他进行对比的数据

        retain_boxes.append(max_element)  # 存储最大的iou元素值

        index = np.where(iou(max_element, other_element_boxes, is_min) < thresh)  # 得到iou比值小于thresh的元素下标
        confidence_boxes = other_element_boxes[index]  # 置信度列表重新存储小于thresh的元素

    if confidence_boxes.shape[0] > 0:
        retain_boxes.append(confidence_boxes[0])

    return np.stack(retain_boxes)


def convert_to_square(bbox):
    square_bbox = bbox.copy()
    if bbox.shape[0] == 0:
        return np.array([])

    # 取宽、高
    w = bbox[:, 2] - bbox[:, 0]
    h = bbox[:, 3] - bbox[:, 1]

    max_side = np.maximum(w, h)
    square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_side * 0.5
    square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - max_side * 0.5
    square_bbox[:, 2] = square_bbox[:, 0] + max_side
    square_bbox[:, 3] = square_bbox[:, 1] + max_side

    return square_bbox


if __name__ == '__main__':
    a = np.array([1, 1, 11, 11])
    bs = np.array([[1, 1, 10, 10], [11, 11, 20, 20]])
    print(iou(a, bs))
    bs = np.array([[1, 1, 10, 10, 0.98], [1, 1, 9, 9, 0.8], [9, 8, 13, 20, 0.7], [6, 11, 18, 17, 0.85]])
    print(-bs[:, 4])
    print((-bs[:, 4]).argsort())
    print(nms(bs))
