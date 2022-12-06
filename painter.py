import cv2
import numpy as np
import time
import os
import cvzone
import mediapipe
from cvzone.HandTrackingModule import HandDetector as TH


def getCount(ar):
    if ar == [0, 0, 0, 0, 0]:
        return 0
    elif ar == [0, 1, 0, 0, 0]:
        return 1
    elif ar == [0, 1, 1, 0, 0]:
        return 2
    elif ar == [0, 1, 1, 1, 0]:
        return 3
    elif ar == [0, 1, 1, 1, 1]:
        return 4
    elif ar == [1, 1, 1, 1, 1]:
        return 5
    elif ar == [0, 1, 0, 0, 1]:
        return 6
    elif ar == [0, 1, 0, 1, 1]:
        return 7
    elif ar == [1, 1, 0, 0, 0]:
        return 8


brush_thickness = 15
eraser_thickness = 100
image_canvas = np.zeros((720, 1280, 3), np.uint8)

currentT = 0
previousT = 0

header_img = "Images"
header_img_list = os.listdir(header_img)
overlay_image = []

for i in header_img_list:
    image = cv2.imread(f'{header_img}/{i}')
    overlay_image.append(image)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

cap.set(cv2.CAP_PROP_FPS, 60)

default_overlay = overlay_image[0]
draw_color = (255, 200, 100)

detector = TH(detectionCon=.85, maxHands=1)

xp = 0
yp = 0

while True:
    imgBG = cv2.imread("Images/bg2.png")
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img[0:125, 0:1280] = default_overlay
    cv2.rectangle(img, (100, 100), (1280 - 100, 720 - 100),
                  (255, 0, 255), 3)

    hands, img = detector.findHands(img)
    if hands:
        if len(hands) == 1:
            hand1 = hands[0]
            lmList1 = hand1["lmList"]
            handType1 = hand1["type"]
            my_fingers = detector.fingersUp(hand1)
            x1, y1 = lmList1[8][0], lmList1[8][1]  # index
            x2, y2 = lmList1[12][0], lmList1[12][1]  # middle
            if getCount(my_fingers) == 2:
                xp, yp = 0, 0
                if y1 < 135:
                    if 400 < x1 < 470:
                        default_overlay = overlay_image[0]
                        draw_color = (255, 0, 0)
                    elif 530 < x1 < 605:
                        default_overlay = overlay_image[1]
                        draw_color = (47, 225, 245)
                    elif 660 < x1 < 745:
                        default_overlay = overlay_image[2]
                        draw_color = (197, 47, 245)
                    elif 820 < x1 < 890:
                        default_overlay = overlay_image[3]
                        draw_color = (53, 245, 47)
                    elif 1100 < x1 < 1280:
                        default_overlay = overlay_image[4]
                        draw_color = (0, 0, 0)

                cv2.putText(img, 'Color Selector Mode', (900, 680), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                            color=(0, 255, 255), thickness=2, fontScale=1)
                cv2.line(img, (x1, y1), (x2, y2), color=draw_color, thickness=3)

            if my_fingers[0] == 0 and my_fingers[2] == 0 and my_fingers[3] == 0 and my_fingers[4] == 0:
                cv2.putText(img, "Drawing Mode", (900, 680), fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(255, 255, 0),
                            thickness=2, fontScale=1)
                cv2.circle(img, (x1, y1), 15, draw_color, thickness=-1)
                if xp == 0 and yp == 0:
                    xp = x1
                    yp = y1
                if draw_color == (0, 0, 0):
                    cv2.line(img, (xp, yp), (x1, y1), color=draw_color, thickness=eraser_thickness)
                    cv2.line(image_canvas, (xp, yp), (x1, y1), color=draw_color, thickness=eraser_thickness)
                else:
                    cv2.line(img, (xp, yp), (x1, y1), color=draw_color, thickness=brush_thickness)
                    cv2.line(image_canvas, (xp, yp), (x1, y1), color=draw_color, thickness=brush_thickness)

                xp, yp = x1, y1

    img_gray = cv2.cvtColor(image_canvas, cv2.COLOR_BGR2GRAY)
    _, imginv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    imginv = cv2.cvtColor(imginv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imginv)
    img = cv2.bitwise_or(img, image_canvas)
    currentT = time.time()
    fps = 1 / (currentT - previousT)
    previousT = currentT

    cv2.putText(img, 'Client FPS:' + str(int(fps)), (10, 670), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                color=(255, 0, 0), thickness=2)

    cv2.imshow('Virtual Painting', img)
    if cv2.waitKey(1) & 0xff == ord(' '):
        break
