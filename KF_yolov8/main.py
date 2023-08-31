import cv2
import datetime
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
from PIL import Image
from KF import KalmanFilter
import math 

import os

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 640)
cap.set(4, 480)
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m','j','p','g'))
# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

#model = YOLO("yolo-Weights/yolov8n.pt")
model = YOLO("yolov8n.pt")
measured_x = 320
measured_y = 240

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

#KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)
KF = KalmanFilter(0.1, 1, 1, 1, 5,0.1)

while True:  
    ret, img = cap.read()
    #frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    results = model(img)
    #annotated_results = results[0].cuda().plot()
    # coordinates
    for r in results:
    #     masks = r.masks
    #     print(masks)
        boxes = r.boxes.cuda()
        obj_detected = False
        for box in boxes:
            cls = int(box.cls[0])
            if classNames[cls] == "cell phone":
                obj_detected = True
                #bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                measured_x = x1 + (x2 - x1)/2
                measured_y = y1 + (y2 - y1)/2
                
                # put box in cam
                cv2.circle(img,  (int(measured_x), int(measured_y)), 10 , (0, 191, 255), 2)
                # confidence
                confidence = math.ceil((box.conf[0]*100))/100
                # object details
                org = [int(measured_x), int(measured_y)]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2
                cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
                break
                #cv2.putText(img, "target", org2, font, fontScale, color, 1)

    KF.update_parameter()
    (pred_x, pred_y) = KF.predict()
    #print(pred_x)
    cv2.rectangle(img, (int(pred_x - 15), int(pred_y - 15)), (int(pred_x + 15), int(pred_y + 15)), (255, 0, 0), 2)
    (est_x1, est_y1) = KF.update(([measured_x] , [measured_y]))
    #print(est_x1)
    cv2.rectangle(img, (int(est_x1 - 15), int(est_y1 - 15)), (int(est_x1 + 15), int(est_y1 + 15)), (0, 0, 255), 2)

    cv2.imshow('Test', img)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()

