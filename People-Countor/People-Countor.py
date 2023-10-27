import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

#cap = cv2.VideoCapture(0)
#cap.set(4, 720)
#cap.set(3, 1280)
cap = cv2.VideoCapture("../Videos/people.mp4")

model = YOLO("../Yolo-Wieghts/yolov8l.pt")

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
mask = cv2.imread("mask.png")

#traking
totalCountup=[]
totalCountdo =[]

tracker =Sort(max_age = 20, min_hits =3, iou_threshold =0.3)
limitsdo = [527, 489, 735, 489]
limitsup = [103, 161, 296, 161]

while True:
    success, img = cap.read()
    imageRegion = cv2.bitwise_and(img, mask)
    imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (730, 260))
    results = model(imageRegion, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:

            #Bounding Box
            x1, y1 , x2, y2 =box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 3)

            w, h = x2-x1, y2-y1

            #Confidence
            conf =math.ceil(box.conf[0] * 100 )/ 100
            print(conf)
            #Class Name
            cls = int(box.cls[0])
            currentClass =classNames[cls]
            if currentClass == "person" and conf > 0.3:
             #   cvzone.putTextRect(img, f"{currentClass} {conf}", (max(0,x1), max(35, y1)), scale=0.7, thickness=1, offset=3)
             #   cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt =5)
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections =np.vstack((detections,currentArray))

    resultsTracker = tracker.update(detections)

    cv2.line(img, (limitsdo[0], limitsdo[1]), (limitsdo[2], limitsdo[3]), (0,0,255), 5)
    cv2.line(img, (limitsup[0], limitsup[1]), (limitsup[2], limitsup[3]), (0, 0, 255), 5)

    for results in resultsTracker:
       x1, y1, x2, y2, Id = results
       x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
       print(results)
       w, h = x2 - x1, y2 - y1
       cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255,0,255))
       cvzone.putTextRect(img, f"{int (Id)}", (max(0, x1), max(35, y1)), scale=1.5, thickness=3, offset=10)

       cx, cy = x1 +w//2, y1 +h//2
       cv2.circle(img ,(cx, cy), 5, (255,0,0), cv2.FILLED)

       if limitsdo[0]< cx < limitsdo[2] and limitsdo[1]-15 <cy <limitsdo[1]+15:
             if totalCountdo.count(Id) ==0:
                 totalCountdo.append(Id)
                 cv2.line(img, (limitsdo[0], limitsdo[1]), (limitsdo[2], limitsdo[3]), (0, 255, 0), 5)
       if limitsup[0] < cx < limitsup[2] and limitsup[1] - 15 < cy < limitsup[1] + 15:
             if totalCountup.count(Id) == 0:
                     totalCountup.append(Id)
                     cv2.line(img, (limitsup[0], limitsup[1]), (limitsup[2], limitsup[3]), (0, 255, 0), 5)

    #cvzone.putTextRect(img, f' Count: {len(totalCountdo)}', (50, 50))
    cv2.putText(img, str(len(totalCountdo)), (1191,345), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)
    cv2.putText(img, str(len(totalCountup)), (929, 345), cv2.FONT_HERSHEY_PLAIN, 5, (0,255,0), 8)
    cv2.imshow("Image", img)
    #  cv2.imshow("ImageRegion", imageRegion)
    cv2.waitKey(1)