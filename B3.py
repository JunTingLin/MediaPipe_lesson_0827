# B3: 在手指根部位置加上透明png圖檔
import cv2
import numpy as np
import mediapipe as mp

def overlay_transparent(background, overlay, overlayX, overlayY):
    """
    overlayX,overlayY為傳統座標系統圖片左上角原點
    """
    background_width = background.shape[1]
    background_height = background.shape[0]  #012高寬深
    if overlayX >= background_width or overlayY >= background_height or overlayX<0 or overlayY<0 : #超出畫面
        return background
    overlayH, overlayW = overlay.shape[0], overlay.shape[1]
    if overlayX + overlayW > background_width:
        overlayW = background_width - overlayX
        overlay=overlay[:,:overlayW]  #高從頭到尾，寬只到overlayW(可參考老師的圖)
    if overlayY + overlayH > background_height:
        overlayH = background_height - overlayY
        overlay = overlay[:overlayH] #高從頭到overlayH
    if overlay.shape[2] < 4:  #假如深度只有RGB三層，沒有alpha
        overlay =np.concatenate( #陣列合併
            [
                overlay,
                np.ones((overlay.shape[0],overlay.shape[1],1),dtype=overlay.dtype) * 255
                #255代表透明，有alpha層
                # 用法: np.empty(shape, dtype, order)
            ],
            axis=2,
        )
    #分離
    overlay_image = overlay[...,:3]  #前面不知道有多少維度用...省略，:3 第一層到第二層
    mask = overlay[...,3:]/255.0 #取出alpha遮罩
    background[overlayY:overlayY+overlayH, overlayX:overlayX+overlayW] = (1.0-mask) * background[overlayY:overlayY+overlayH, overlayX:overlayX+overlayW]\
                                                                         + mask  *  overlay_image
    # mask=0代表使用原background(frame)的底圖
    return background


overlay = cv2.imread('logo95.png',cv2.IMREAD_UNCHANGED)
# image read讀進來的資料，會儲存成一個 NumPy 的陣列
#IMREAD_UNCHANGED 讀取圖片中所有的 channels，包含透明度的 channel
print("讀取到圖片，圖片維度:",overlay.shape)

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
                    cx, cy = int(lm.x * w), int(lm.y * h)  # 圓心 #換成成傳統座標系
                    if id==20 or id==16 or id==12 or id==8 or id==4:  #指尖的特徵點
                        cx, cy = cx-int(overlay.shape[1]/2), cy-int(overlay.shape[0]/2)  #看圖，圓心往左上方移動
                        overlay_transparent(frame,overlay,cx,cy)
                    else:
                        cv2.circle(frame,(cx,cy),3,(0,0,0),cv2.FILLED) #非指頭特徵點，圓心給我塗黑
        cv2.imshow('Lin JJ',frame)
        if cv2.waitKey(100) & 0xFF == 27:
            break
    else:
        continue
cap.release()
cv2.destroyAllWindows()

