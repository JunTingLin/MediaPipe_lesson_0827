import cv2
import pafy
import mediapipe as mp

url="https://www.youtube.com/watch?v=RMWGZEgigbA"
live = pafy.new(url)
stream = live.getbest(preftype="mp4")

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2,min_detection_confidence=0.5,min_tracking_confidence=0.5)
cap = cv2.VideoCapture(stream.url)
while cap.isOpened():
    success, frame = cap.read()
    if success:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imshow('Lin JJ', frame)
        if cv2.waitKey(100) & 0xFF == 27:
            break
    else:
        continue
cap.release()
cv2.destroyAllWindows()