import logging
import os
import requests
import argparse
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import os
from datetime import datetime
from time import time
import imutils
from models.common import DetectMultiBackend
from utils.general import ( check_img_size,non_max_suppression, scale_coords,LOGGER)
from utils.torch_utils import select_device, time_sync
from utils.augmentations import letterbox
from utils import all_utils
from model import model_load

FRAMES_TO_PERSIST = 10 # number of frames after compare
MIN_SIZE_FOR_MOVEMENT = 20000000 # movement size
MOVEMENT_DETECTED_PERSISTENCE = 100


def preprocess_image(im, half, device):
    """it will preprocess image for model 

    Args:
        im ([image]): [description]
        half ([type]): [description]
        device ([type]): [description]

    Returns:
        image: [description]
    """
    im = letterbox(im,640,32)[0] 
    im = im.transpose((2, 0, 1))[::-1]
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im).to(device)
    im = im.half() if half else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    
    return im

def filter(det):
    """this function will filter out gurad from prediction

    Args:
        det ([type]): [description]

    Returns:
        tensor: [description]
    """
    new = [t for t in det if t[-1]==0]
    try:
        new_t=torch.stack(new)
    except:
        new_t = torch.tensor([]).to("cpu")
    return new_t
@torch.no_grad()
def run(weights="",
        source = "s.mp4",  # model.pt path(s)
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    FRAMES_TO_PERSIST = 10 # number of frames after compare
    MIN_SIZE_FOR_MOVEMENT = 20000 # movement size
    MOVEMENT_DETECTED_PERSISTENCE = 100    
    first_frame = None
    next_frame = None
    font = cv2.FONT_HERSHEY_SIMPLEX
    delay_counter = 0
    movement_persistent_counter = 0

    model,device = model_load(device=device,weights=weights,dnn=dnn,imgsz=imgsz,half=half)
    cap = cv2.VideoCapture(source)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out_cam=cv2.VideoWriter("output.mp4",0x00000021, 10, (frame_width,frame_height))
    try:
        while True:
            _,frame = cap.read()
            im = frame.copy()
            im0 = frame.copy()
            im0s = frame.copy()
            im = preprocess_image(im,half,device)
            pred = model(im,augment=augment,visualize=False)

            pred = non_max_suppression(pred,conf_thres,iou_thres,classes,agnostic_nms,max_det=max_det)

            for det in pred:
                det = filter(det)
                if len(det):
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    for *xyxy, conf, cls in reversed(det):
                        c1,c2 = (int(xyxy[0]),int(xyxy[1])), (int(xyxy[2]),int(xyxy[3]))
                        img12 = im0[c1[1]:c2[1],c1[0]:c2[0]]
                        frame = imutils.resize(img12,width = 350)
                        frame = cv2.resize(frame, (350,250))
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        gray = cv2.GaussianBlur(gray, (21, 21), 0)
                        if first_frame is None: first_frame = gray
                        delay_counter += 1
                        if delay_counter > FRAMES_TO_PERSIST:
                            delay_counter = 0
                            first_frame = next_frame
                        next_frame = gray
                        frame_delta = cv2.absdiff(first_frame, next_frame)
                        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
                        thresh = cv2.dilate(thresh, None, iterations = 2)
                        cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        transient_movement_flag = False
                        for cn in cnts:
                            (x, y, w, h) = cv2.boundingRect(cn)
                            print("-"*50)
                            print(cv2.contourArea(cn))
                            transient_movement_flag = cv2.contourArea(cn) > MIN_SIZE_FOR_MOVEMENT

                        if transient_movement_flag:
                            movement_persistent_counter = MOVEMENT_DETECTED_PERSISTENCE

                        if movement_persistent_counter > 0:
                            text = "Guard: " + "not sleeping"
                            movement_persistent_counter -= 1
                        else: # No movement detected
                            text = "Guard :- sleeping"
                        cv2.putText(im0, str(text), (10,35), font, 0.75, (0,0,255), 2, cv2.LINE_AA)
                        frame_delta = cv2.cvtColor(frame_delta, cv2.COLOR_GRAY2BGR)
                        cv2.rectangle(im0,c1,c2,color=(0,0,255))
                        cv2.putText(im0, "guard", (int(xyxy[0]),int(xyxy[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

                else:
                    pass
                out_cam.write(im0)
                cv2.imshow("frame",im0)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except:
        print("complete")
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default= os.path.join(os.getcwd(),"yolov5s.pt"), help='model path(s)')
    parser.add_argument('--source', type=str, default='s.mp4', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt

def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    print("Inside Main")
    
    opt = parse_opt()
    main(opt)
