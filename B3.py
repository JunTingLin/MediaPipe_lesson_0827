# -*- coding: utf-8 -*-
import cv2
import numpy as np
import mediapipe as mp

def overlay_transparent(background, overlay, overlayX, overlayY):
    background_width = background.shape[1]
    background_height = background.shape[0]  #012高寬深
    if overlayX >= background_width or overlayY >= background_height: #超出畫面
        return background
    overlayH, overlayW = overlay.shape[0], overlay.shape[1]
    if overlayX + overlayW > background_width:
        overlayW = background_width - overlayX
        overlay=overlay[:,:overlayW]  #高從頭到尾，寬只到overlayW
    if overlayY + overlayH > background_height:
        overlayH = background_height - overlayY
        overlay = overlay[:overlayH] #高從頭到overlayH
    if overlay.shape[2] < 4:  #只有RGB，沒有alpha
        overlay =np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0],overlay.shape[1],1),dtype=overlay.dtype) * 255 #255代表透明，有alpha層
            ],
            axix=2,
        )
    overlay_image = overlay[...,:3]  #前面不知道有多少維度用...省略，:3 第一層到第二層
    mask = overlay[...,3:]/255.0
    background[overlayY:overlayY+overlayH, overlayX:overlayX+overlayW] = (1.0-mask)*background[overlayY:overlayY+overlayH, overlayX:overlayX+overlayW]+mask*overlay_image
    # mask=0代表完全透明
    return background


overlay = cv2.imread('logo95.png',cv2.IMREAD_UNCHANGED)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=10,min_detection_confidence=0.5,min_tracking_confidence=0.5)
cap=cv2.VideoCapture(0)
# BGR
drawing_space1=mp_drawing.DrawingSpec(color=(127,255,0),thickness=10,circle_radius=10) #偏綠色
drawing_space2=mp_drawing.DrawingSpec(color=(0,127,255),thickness=3,circle_radius=5) #橘子色

while (cap.isOpened()):
    success, frame=cap.read()
    if success:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, c = image.shape
        results=hands.process(image)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          landmark_drawing_spec=drawing_space2, connection_drawing_spec=drawing_space1)
                for id,lm in enumerate(hand_landmarks.landmark): #id為第幾號特徵點
                    cx, cy = int(lm.x * w), int(lm.y * h)  # 圓心 #傳統座標系統轉換成MediaPipe座標
                    if id==20 or id==16 or id==12 or id==8 or id==4:
                        cx, cy = cx-int(overlay.shape[1]/2), cy-int(overlay.shape[0]/2)  #logo中央放在圓心
                        overlay_transparent(frame,overlay,cx,cy)
                    else:
                        cv2.circle(frame,(cx,cy),3,(0,0,0),cv2.FILLED)
        cv2.imshow('Lin JJ',frame)
        if cv2.waitKey(100) & 0xFF == 27:
            break
    else:
        continue
cap.release()
cv2.destroyAllWindows()
