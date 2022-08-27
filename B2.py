# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=10,min_detection_confidence=0.5,min_tracking_confidence=0.5)
cap=cv2.VideoCapture(0)
# BGR
drawing_space1=mp_drawing.DrawingSpec(color=(127,255,0),thickness=10,circle_radius=10) #偏綠色
drawing_space2=mp_drawing.DrawingSpec(color=(0,127,255),thickness=3,circle_radius=5) #橘子色

while cap.isOpened():
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
                    cx,cy = int(lm.x*w), int(lm.y*h)  #圓心 #傳統座標系統轉換成MediaPipe座標
                    cv2.circle(frame,(cx,cy),3,(0,255,255),cv2.FILLED)  #黃色
        cv2.imshow('Lin JJ',frame)
        if cv2.waitKey(100) & 0xFF == 27:
            break
    else:
        continue
cap.release()
cv2.destroyAllWindows()


