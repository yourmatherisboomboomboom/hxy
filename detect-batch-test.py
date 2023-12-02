#coding:utf-8

import argparse
import os
import sys
import time
from pathlib import Path

import glob
import hashlib
import json
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from threading import Thread
from zipfile import ZipFile

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import ExifTags, Image, ImageOps
from torch.utils.data import DataLoader, Dataset, dataloader, distributed
from tqdm import tqdm

from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
from utils.general import (DATASETS_DIR, LOGGER, NUM_THREADS, check_dataset, check_requirements, check_yaml, clean_str,
                           segments2boxes, xyn2xy, xywh2xyxy, xywhn2xyxy, xyxy2xywhn)
from utils.torch_utils import torch_distributed_zero_first

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadStreams, LoadImagesAndLabels, _RepeatSampler, LoadImages, \
    InfiniteDataLoader, LoadImagesBatch
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

# YOLOv5默认阈值
conf_th = 0.6
iou_th = 0.01
model_path= './runs/best.pt'
# 图像路径
# image_file = "dataset/23fz/after/3[民国26年]重修邵武县志.tif"  # 替换为您的实际图像路径
image_file = "dataset/images_org/GZSB192411110005.tif"  # 替换为您的实际图像路径
# image_file = "a13c90a1-1be1-9a76-c545-6292f9c63e2bA002__006.tif"  # 替换为您的实际图像路径
# image_file = "/home/141Dpan/OCR文件夹/零散图书扫描/BZ202305120001/"  # 替换为您的实际图像路径
device = 'cuda:0'if torch.cuda.is_available() else 'cpu'
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
s=time.time()
device = select_device(device)
model = DetectMultiBackend(model_path, device=device, dnn=False, data='dataset/coco128.yaml')
stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine

# print('Loading Time:', time.time() - s)

prefix='Detect:'
import os
os.environ["OMP_NUM_THREADS"] = "1"


def map_predictions_to_original(image, dir, pred,sub,split):
    if split==0:
        return pred
    img_height, img_width, _ = image.shape
    pred_mapped = []
    BigCnt = sub[0].item()
    for i in range(len(pred)):
        sub_img_dir = dir[i]
        sub_img_pred = pred[i]
        # if i<len(pred)-2:
        if i<len(pred)-BigCnt:
            # if i<len(pred)-BigCnt-2:
            #     continue
            # 计算子图在原图中的位置
            x, y, x2, y2 = sub_img_dir
            x = int(x)
            y = int(y)

            sub_img_pred[:, 0] += x
            sub_img_pred[:, 1] += y
        else:
            x, y, x2, y2 = sub_img_dir
            img_width = x2 - x
            img_height = y2 - y

            sub_img_pred[:, 0] *= img_width/(dir[0][2]-dir[0][0])
            sub_img_pred[:, 1] *= img_height/(dir[0][3]-dir[0][1])
            #w*h
            sub_img_pred[:, 2] *= img_width/(dir[0][2]-dir[0][0])
            sub_img_pred[:, 3] *= img_height/(dir[0][3]-dir[0][1])


            sub_img_pred[:, 1] += y
            sub_img_pred[:, 0] += x
        pred_mapped.append(sub_img_pred.cpu())

    # 将预测结果转化为一张图片的格式
    pred_mapped = np.concatenate(pred_mapped, axis=0)
    pred_mapped = pred_mapped.reshape([1,-1, 6])

    return torch.from_numpy(pred_mapped).to(device)


def create_dataloader(path, imgsz, workers=1,shuffle=False,split=0):
    dataset = LoadImagesBatch(path, imgsz,split)
    batch_size = len(dataset)
    nw = workers
    sampler = None
    loader = InfiniteDataLoader  # only DataLoader allows for attribute updates
    return loader(dataset,
                  batch_size=batch_size,
                  shuffle=shuffle and sampler is None,
                  # num_workers=nw,
                  sampler=sampler,
                  pin_memory=True), dataset


@torch.no_grad()
def run(
        source=image_file,  # file/dir/URL/glob, 0 for webcam
        imgsz=(320,320),  # inference size (height, width)
        conf_thres=conf_th,  # confidence threshold
        iou_thres=iou_th,  # NMS IOU threshold
        max_det=1000000000,  # maximum detections per image
        device= device,  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnost
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=True,  # hide labels
        hide_conf=True,  # hide confidences
        half=True,  # use FP16 half-precision inference
        split=1,
        saveTo=''
        ):
    if split==0 or split==1:
        imgsz=[832,832]
    source = str(source)
    # print('URL:',source)
    # print(conf_thres,iou_thres)
    model.model.float()

    bs=1
    val_loader, dataset = create_dataloader(source, imgsz,shuffle=False,split=split)


    # Run inference
    # model.warmup(imgsz=(1 if pt else bs, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    res=[]
    # print("Loading")

    for path, image, im, im0s, dir, sub_image_size in val_loader:

        t1 = time_sync()
        im = im.to(device)
        im = im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        pred = model(im)
        t3 = time_sync()
        dt[1] += t3 - t2
        if split==0:
            #mns
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        #先恢复成样本之大小，去变形以及padding
        for i, det in enumerate(pred):
            pred[i][:, :4] = scale_coords(im.shape[2:],pred[i][:, :4], im0s[i].shape).round()

        if split!=0:
            # 映射回原图的坐标
            pred = map_predictions_to_original(dataset.image, dataset.dir, pred,sub_image_size,split)
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3


        det=pred[0]
        seen += 1
        p, im0, frame = path, dataset.image.copy(), getattr(dataset, 'frame', 0)
        annotator = Annotator(im0, line_width=line_thickness, example=str(names))
        if len(det):
            # Write results
            for index,(*xyxy, conf, cls) in enumerate(reversed(det)):

                c = int(cls)  # integer class
                label = None if hide_labels else (names[c] if hide_conf else f' {index}__{conf:.2f}')
                annotator.box_label(xyxy, label, color=colors(c, True))

            # Stream results
            im0 = annotator.result()
            # Print results
            t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
            # LOGGER.info(
            #     f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
            # res.append(im0)

            if not os.path.exists('output-1/'+f'{saveTo}'):
                os.makedirs('output-1/'+f'{saveTo}')  # 如果文件夹不存在，创建它
                print("Create:",'output-1/'+f'{saveTo}')
            # image_path = os.path.join('output-1/'+f'{saveTo}', f'output_{image_file[len("/home/141Dpan/OCR文件夹/零散图书扫描/BZ202305120001/"):]}.jpg')
            image_path = os.path.join('output-1/'+f'{saveTo}', f'output_{image_file[len("/home/Yolov5_Kungyu/dataset/23fz/after"):]}.jpg')
            cv2.imwrite(image_path, im0)
            print("save to :",image_path)


if __name__ == "__main__":

    # path = "/home/141Dpan/OCR文件夹/零散图书扫描/"
    path = "/home/Yolov5_Kungyu/dataset/23fz/"
    tlt = [
        'after'
    # "BZ202305120001",
    # "BZ202305120002",
    # "BZ202305120003",
    # "BZ202305120004",
    # "BZ202305120005",
    # "BZ202305120006",
    # "BZ202305120007",
    # "BZ202305120008",
    # "BZ202305120009",
    # "BZ202305120010",
    # "BZ202305120011",
    # "BZ202305120012",
    # "BZ202305120013",
    # "BZ202305120014",
    # "BZ202305120015"
    ]
    for i in tlt:
        folder_path = path+i  # 文件夹路径
        tif_files = glob.glob(os.path.join(folder_path, "*.*"))  # 匹配所有以.tif结尾的文件
        for tif_file in tqdm(tif_files):
            image_file=tif_file
            run(source=image_file,saveTo=i,split=1)
