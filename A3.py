# A3: 播YouTube直播
import cv2
import pafy
url="https://www.youtube.com/watch?v=UCG1aXVO8H8"
live = pafy.new(url)
stream = live.getbest(preftype="mp4")
cap = cv2.VideoCapture(stream.url)
while(True):
    success, frame = cap.read()
    frame = cv2.resize(frame,(960,540))
    cv2.imshow('AI20220827',frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()