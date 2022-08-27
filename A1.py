import cv2

cap = cv2.VideoCapture(0)

while(True):
    success, frame = cap.read()

    cv2.imshow('AI20220827',frame)
    if cv2.waitKey(1) & 0xFF == 27: # waitKey設定每千分之一秒讀取鍵盤，按esc鍵跳開
        break
cap.release()
cv2.destroyAllWindows()


