# C1: 臉部動作
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=3,min_detection_confidence=0.5,min_tracking_confidence=0.5)
cap=cv2.VideoCapture(0)
# BGR
drawing_space1=mp_drawing.DrawingSpec(color=(0,255,0),thickness=1,circle_radius=1) #綠色
drawing_space2=mp_drawing.DrawingSpec(color=(0,0,255),thickness=1,circle_radius=1) #紅色

while (cap.isOpened()):
    success, frame=cap.read()
    if success:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results=face_mesh.process(image)
        if results.multi_face_landmarks:
            for id, face_landmarks in enumerate(results.multi_face_landmarks):
                if id%2 ==0:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=drawing_space1,
                        connection_drawing_spec=drawing_space1
                    )
                else:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=drawing_space2,
                        connection_drawing_spec=drawing_space2
                        #是基數張臉的用綠色畫臉特徵點跟骨架
                    )
        cv2.imshow('Lin JJ',frame)
        if cv2.waitKey(100) & 0xFF == 27:
            break
    else:
        continue
cap.release()
cv2.destroyAllWindows()


