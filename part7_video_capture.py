import time
import torch
import cv2
from PIL import Image
from part6_detect import Detector

with torch.no_grad() as grad:
    detector = Detector("Data/Params_/P_max/p_net_50.pth",
                        "Data/Params_2/R_max/r_net_19.pth",
                        "Data/Params_2/O_max/o_net_36.pth")

    cap = cv2.VideoCapture(0)
    while True:
        # 逐帧捕获
        x = time.time()
        ret, frame = cap.read()
        # 如果正确读取帧，ret为True
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        boxes_ = detector.detect(img)
        print(boxes_.shape)
        for box in boxes_:
            _x11 = int(box[0])
            _y11 = int(box[1])
            _x22 = int(box[2])
            _y22 = int(box[3])
            cv2.rectangle(frame, (_x11, _y11), (_x22, _y22), (0, 0, 255), 3)
        if not ret:
            break
        # 显示结果帧e
        cv2.imshow('frame', frame)
        print(time.time() - x)
        if cv2.waitKey(1) == ord('q'):
            break

    # 完成所有操作后，释放捕获器
    cap.release()
    cv2.destroyAllWindows()
