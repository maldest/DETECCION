import cv2
from ultralytics import YOLO

#seleccion de camara

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 720)

#ruta modelo
model = YOLO('best.pt')

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo capturar el frame")
        break

    #nivel de confianza
    results = model(frame, imgsz=640, conf=0.8)

    annotated_frame = results[0].plot()

    cv2.imshow('Detección de basura', annotated_frame)

    if cv2.waitKey(1) == 27: #tecla esc
        break


cap.release()
cv2.destroyAllWindows()
