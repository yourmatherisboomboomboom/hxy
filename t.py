# coding=utf-8

'''
#
#    @ FILE: detect_now.py.py
#    @ TIME: 2022/3/11 12:54 PM

#   Created by KungyuHsu🍺 on 2022/3/11 .
#   Copyright 2022 KungyuHsu. All rights reserved.
#
'''
import cv2
import onnx
import numpy as np
from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
import torch
from utils.general import non_max_suppression,scale_coords
import json
import time
dic=['床', '桌椅', '油烟机', '柜子', '马桶', '冰箱', '水池', '沙发', '洗衣机']

class YOLO:
    def __init__(self,weights='yolov5n.pt',device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),data='./models/yolov5n.yaml'):
        '''
        :param weights: weight path
        :return:
        '''
        model=DetectMultiBackend(weights, device=device, data=data)
        model.model.float()
        self.model=model

    def output(self,img0,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),conf_thres=0.25,iou_thres=0.45,classes=None,v=False):
        '''
        :param img: cv2
        :return:
        '''
        img = letterbox(img0, 320, stride=32, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]
        im = np.ascontiguousarray(img)
        im = torch.from_numpy(im).to(device)
        im = im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]
        pred = self.model(im)
        return pred
        # pred = non_max_suppression(pred, conf_thres, iou_thres, classes, max_det=1000)[0]
        # res=[]
        # # s=img0.shape[0:2]
        # # img0=cv2.resize(img0,int(y/2),int(x/2))
        # pred[:, :4] = scale_coords(im.shape[2:], pred[:, :4], img0.shape).round()
        # for x in pred:
        #     posi=[int(k) for k in x[:4]]
        #     Single={
        #         'type': dic[int(x[-1])],
        #         'conf': round(float(x[-2])*100,2),
        #         'posi': [int(k) for k in x[:4]]
        #     }
        #     res.append(Single)
        #     if v:
        #         cv2.rectangle(img0, (posi[0],posi[1]),(posi[2],posi[3]),(0,255,0), 2)
        # if v:
        #     # cv2.imshow('res',img0)
        #     # cv2.waitKey(0)
        #     cv2.imwrite('./game/rooms/update.jpg',img0)
        # return res
    def input(self,img0,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),conf_thres=0.25,iou_thres=0.45,classes=None,v=False):
        '''
        :param img: cv2
        :return:
        '''
        img = letterbox(img0, 320, stride=32, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]
        im = np.ascontiguousarray(img)
        im = torch.from_numpy(im).to(device)
        im = im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]
        return im
        # pred = self.model(im)
        # pred = non_max_suppression(pred, conf_thres, iou_thres, classes, max_det=1000)[0]
        # res=[]
        # # s=img0.shape[0:2]
        # # img0=cv2.resize(img0,int(y/2),int(x/2))
        # pred[:, :4] = scale_coords(im.shape[2:], pred[:, :4], img0.shape).round()
        # for x in pred:
        #     posi=[int(k) for k in x[:4]]
        #     Single={
        #         'type': dic[int(x[-1])],
        #         'conf': round(float(x[-2])*100,2),
        #         'posi': [int(k) for k in x[:4]]
        #     }
        #     res.append(Single)
        #     if v:
        #         cv2.rectangle(img0, (posi[0],posi[1]),(posi[2],posi[3]),(0,255,0), 2)
        # if v:
        #     # cv2.imshow('res',img0)
        #     # cv2.waitKey(0)
        #     cv2.imwrite('./game/rooms/update.jpg',img0)
        # return res



# if __name__ == '__main__':
#     yolo=YOLO()
#     mdl=yolo.model
#     mdl.eval()
#     img = cv2.imread("update.jpg")
#     print(yolo.output(img).shape)
#     print(yolo.input(img).shape)
#     torch.onnx.export(
#         mdl,
#         yolo.input(img),
#         'uhome.onnx',
#         opset_version=11
#         # input_names=["input_0"],
#         # output_names=["output_0"],
#         # dynamic_axes={'input_0': [0], 'output_0': [0]}
#     )
#     from onnxruntime.quantization import quantize_dynamic, QuantType
#
#     quantized_model = quantize_dynamic('uhome.onnx', "uhome_model_quant.onnx", weight_type=QuantType.QUInt8)


# if __name__ == '__main__':
#     yolo=YOLO()
#     img=cv2.imread("update.jpg")
#     t=time.time()
#     res=yolo(img,v=True)
#     print('time:',time.time()-t)
#     t=time.time()
#     print(json.dumps(res,ensure_ascii=False))

if __name__ == '__main__':
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantized_model = quantize_dynamic('yolov5n.onnx', "y5n_model_quant.onnx", weight_type=QuantType.QUInt8)