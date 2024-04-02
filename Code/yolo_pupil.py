import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO("pupil_detector.pt")
# model.export(format='onnx')


webcam = cv2.VideoCapture(0)

while 1:
    ret, frame = webcam.read()

    results = model.predict([frame])

    box = np.int32(np.round(results[0].boxes.xyxy)).tolist()
    for b in box:
        cv2.rectangle(frame, (b[0],b[1]), (b[2],b[3]), (0,0,255),1)
    cv2.imshow("eyes", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

webcam.release()
cv2.destroyAllWindows()
