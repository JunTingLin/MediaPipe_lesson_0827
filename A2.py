import cv2

cap = cv2.VideoCapture('20220827.mp4')
if(cap.isOpened()==False):
    print("Error opening video stream or file")

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        cv2.imshow('AI20220827',frame)
        if cv2.waitKey(25) & 0xFF == 27:
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()