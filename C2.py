# C2: 將臉部嘴唇替換成吸血鬼圖檔
import cv2
import numpy as np
import mediapipe as mp
def overlay_transparent(background, overlay, overlayX, overlayY):
    background_width = background.shape[1]
    background_height = background.shape[0]
    if overlayX >= background_width or overlayY >= background_height:
        return background
    overlayH, overlayW = overlay.shape[0], overlay.shape[1]
    if overlayX + overlayW > background_width:
        overlayW = background_width - overlayX
        overlay = overlay[:, :overlayW]
    if overlayY + overlayH > background_height:
        overlayH = background_height - overlayY
        overlay = overlay[:overlayH]
    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )
    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0
    background[overlayY:overlayY+overlayH, overlayX:overlayX+overlayW] = (1.0 - mask) *\
        background[overlayY:overlayY+overlayH, overlayX:overlayX+overlayW] + mask * overlay_image
    return background

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=10,min_detection_confidence=0.5,min_tracking_confidence=0.5)
mouth_normal=cv2.imread("lips2.png",cv2.IMREAD_UNCHANGED)
cap=cv2.VideoCapture(0)
# BGR
drawing_space1=mp_drawing.DrawingSpec(color=(0,255,0),thickness=1,circle_radius=1) #偏綠色
drawing_space2=mp_drawing.DrawingSpec(color=(0,0,255),thickness=1,circle_radius=1) #橘子色

while (cap.isOpened()):
    success, frame=cap.read()
    if success:
        try:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h,w,d=frame.shap
            results=face_mesh.process(image)
            if results.multi_face_landmarks:
                # 讀出一張一張臉，每張臉順序id編號
                for id, face_landmarks in enumerate(results.multi_face_landmarks):
                    mouth_h=int((face_landmarks.landmark[17].y*h)-(face_landmarks.landmark[0].y*h))
                    mouth_w=int((face_landmarks[287].x*w)-face_landmarks.landmark[57].x*w)
                    mouth = cv2.resize(mouth_normal,(mouth_w,mouth_h))
                    x,y =int(face_landmarks.landmark[57].x*w), int(face_landmarks.landmark[0].y*h)
                    overlay_transparent(frame, mouth,x,y)
            cv2.imshow('Lin JJ',frame)
            if cv2.waitKey(100) & 0xFF == 27:
                break
        except:
            pass
    else:
        continue
cap.release()
cv2.destroyAllWindows()