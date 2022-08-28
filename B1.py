# B1: 手部動作
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=3,min_detection_confidence=0.5,min_tracking_confidence=0.5)
cap=cv2.VideoCapture(0)
while cap.isOpened():
    success, frame=cap.read()
    #frame = cv2.resize(frame, (960, 540))  # 更改frame大小
    if success:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV較怪，要將讀入的每張影格BGR格式轉成RGB存入image變數
        print(image.shape) #(預設長640、寬480、深3層)顯示(480,640,3)
        results=hands.process(image) #mediapipe使用一般RGB，但格子座標系統需要正規化
        if results.multi_hand_landmarks:
            print("測到幾隻手:",len(results.multi_hand_landmarks))
            for hand_landmarks in results.multi_hand_landmarks:
                print("手部特徵點數:",len(hand_landmarks.landmark)) #手部有21個特徵點
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imshow('Lin JJ',frame)
        if cv2.waitKey(100) & 0xFF == 27:
            break
    else:
        continue
cap.release()
cv2.destroyAllWindows()


