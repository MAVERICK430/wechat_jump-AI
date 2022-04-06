import argparse
import os
import sys
from pathlib import Path
import numpy as np
import threading
import math

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
from utils.general import (DATASETS_DIR, LOGGER, NUM_THREADS, check_dataset, check_requirements, check_yaml, clean_str,
                           segments2boxes, xyn2xy, xywh2xyxy, xywhn2xyxy, xyxy2xywhn)
from utils.torch_utils import torch_distributed_zero_first

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from grabscreen import grab_screen
from ctypes import *
import time

# os.add_dll_directory("D:/biancheng/code/Me/python/yolov5/TYT/Dll/")

mouse = windll.LoadLibrary('./Dll/DllDemo.dll')
# mouse = CDLL('User32.dll', winmode=0)
# mouse = CDLL('D:/biancheng/code/Me/python/yolov5/TYT/Dll/mouse.dll', winmode=0)

device = select_device('')
weights = './Files/tyt_250img.onnx'
data = './Files/tyt_data.yaml'
imgsz = (640, 640)

model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size(imgsz, s=stride)  # check image size

# 加载图片
# dataset = LoadImages('', img_size=imgsz, stride=stride, auto=pt)
exitFlag = 0
tytJuli = 0


def gettytJuli():
    return tytJuli


def settytJuli(data):
    global tytJuli
    tytJuli = int(data)


class myThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        print("开始线程")
        Run_tyt()
        print("退出线程")


def Run_tyt():
    while True:
        if exitFlag == 1:
            break
        val = gettytJuli()
        if val != 0:
            # print("开始跳跃")
            mouse.click(gettytJuli())
            # print("结束跳跃")
            settytJuli(0)
            print(gettytJuli())
            time.sleep(2.9)


# 启动点击
thread1 = myThread()
thread1.start()

# 截图大小
y_towindow = 200
height_towindow = 450
width_towindow = 450

width_show = 0
height_show = 0
click_show = 0
while True:
    img0 = grab_screen(region=(0, y_towindow, width_towindow, height_towindow))
    start_time = time.time()

    img = letterbox(img0, imgsz, stride=stride, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)

    bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0

    # 推理
    t1 = time_sync()
    img = torch.from_numpy(img).to(device)
    img = img.half() if model.fp16 else img.float()  # uint8 to fp16/32
    img /= 255  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    t2 = time_sync()
    dt[0] += t2 - t1

    # Inference
    pred = model(img, augment=False, visualize=False)
    t3 = time_sync()
    dt[1] += t3 - t2

    # NMS
    pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)
    dt[2] += time_sync() - t3

    for i, det in enumerate(pred):  # per image
        seen += 1

        im0 = img0

        # s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        imc = im0.copy() if False else im0  # for save_crop
        annotator = Annotator(im0, line_width=3, example=str(names))
        if len(det):

            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                # s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            name = []
            xy = []
            for *xyxy, conf, cls in reversed(det):

                if True:  # Write to file
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    if conf < 0.4:
                        print('置信度太低！！')
                        continue
                    name.append(int(cls))
                    xy.append(xywh)
                    line = (cls, *xywh, conf) if False else (cls, *xywh)  # label format

                if True:  # Add bbox to image
                    c = int(cls)  # integer class
                    label = None if False else (names[c] if False else f'{names[c]} {conf:.2f}')
                    # annotator.box_label(xyxy, str(round(xywh[0], 4)), color=colors(c, True))
                    annotator.box_label(xyxy, label, color=colors(c, True))

            # print(name)
            # print(xy)

            index = 0
            for key in name:
                if int(key) == 0:
                    break
                else:
                    index = index + 1

            try:
                # 取小人的信息
                data = xy[index]
                # 筛选出比小人坐标低的
                xy_index = 0
                # 判断出来的最佳对象
                xy_data = [1, 2, 3]
                # print(xy_index)
                # print(index)
                # print(name)
                # print(xy)
                for data_xy in xy:
                    # print(data_xy)
                    if xy_index == index:
                        # 这里不能用continue
                        a = 1

                    # print(data_xy[1])
                    # print(data[1])
                    # print(data_xy[1] < data[1])
                    if data_xy[1] < data[1]:
                        xy_data = data_xy
                        break
                    xy_index = xy_index + 1

                juli = xy_data[0] - data[0]

                if juli < 0:
                    juli = -juli

                # 获取距离后 点击固定时常 然后松开  程序暂停1秒

                # print(data)
                # print(xy_data)

                # x
                xiaorenx = data[0]
                # y
                xiaoreny = data[1]
                # width
                xiaorenWidth = data[2]
                # height
                xiaorenheight = data[3]

                # x
                FangKuaix = xy_data[0]
                # y
                FangKuaiy = xy_data[1]
                # width
                FangKuaiWidth = xy_data[2]
                # height
                FangKuaiheight = xy_data[3]

                # print('没到这？')

                # print(xiaorenx + xiaorenWidth / 2)
                # print(xiaoreny + xiaorenheight / 2)
                # print(FangKuaix + FangKuaiWidth / 2)
                # print(FangKuaiy + FangKuaiheight / 2)
                width = 0
                height = 0

                # 右边
                if xiaorenx + xiaorenWidth / 2 < FangKuaix + FangKuaiWidth / 2:
                    # 带宽度
                    # width = xiaorenx + xiaorenWidth / 2 - FangKuaix + FangKuaiWidth / 2
                    # height = xiaoreny + xiaorenheight / 2 - FangKuaiy + FangKuaiheight / 2

                    width = xiaorenx - FangKuaix
                    height = xiaoreny + xiaorenheight / 2 - FangKuaiy
                    # 左边
                else:
                    # 带宽度
                    # width = FangKuaix + FangKuaiWidth / 2 - xiaorenx + xiaorenWidth / 2
                    # height = xiaoreny + xiaorenheight / 2 - FangKuaiy + FangKuaiheight / 2

                    width = FangKuaix - xiaorenx
                    height = xiaoreny + xiaorenheight / 2 - FangKuaiy

                if width < 0:
                    width = - width
                if height < 0:
                    height = -height

                width_show = width
                height_show = height

                # 显示位置
                cv2.putText(im0, str('local'),
                            (int(xiaorenx * width_towindow),
                             int((xiaoreny + xiaorenheight / 2) * height_towindow)),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 34, 170), 3)

                cv2.putText(im0, str('box'),
                            (int((FangKuaix * width_towindow)),
                             int((FangKuaiy * height_towindow))),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 34, 170), 3)

                tan = math.pow(width, 2) + math.pow(height, 2)

                weitiao = 2.45
                cheng = math.pow(width_towindow, 2) + math.pow(height_towindow, 2)
                cheng = math.sqrt(cheng)
                # cheng = 1000

                # print(cheng)

                # print(int((math.sqrt(tan) * 1000) * weitiao))

                tytJuli = int((math.sqrt(tan) * cheng) * weitiao)
                click_show = tytJuli

                # print(tytJuli)


            except Exception as e:
                a = 1
                # pass
                print('str(e):\t\t', str(e))

        cv2.putText(im0, str(width_show), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 34, 170), 3)
        cv2.putText(im0, str(height_show), (0, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 34, 170), 3)
        cv2.putText(im0, 'click:' + str(click_show) + 'ms',
                    (0, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 34, 170), 3)

        # Stream results
        im0 = annotator.result()
        if True:
            cv2.namedWindow('yolo_tyt', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('yolo_tyt', int(width_towindow // 1.5), int(height_towindow // 1.5))
            cv2.imshow('yolo_tyt', im0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
            # cv2.waitKey(1)  # 1 millisecond

    # LOGGER.info(f'Done. ({t3 - t2:.3f}s)')
    print(1000 / int(1000 * (time.time() - start_time)))
