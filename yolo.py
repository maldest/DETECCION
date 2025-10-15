from ultralytics import YOLO
from multiprocessing import freeze_support

def main():
    #Carga el modelo base
    model = YOLO('models/yolov8n.pt')

    #Entrenamiento
    model.train(
        data='dataTrain/data.yaml',
        epochs=50,
        batch=32,
        imgsz=640,
        device='cuda'
    )

if __name__ == '__main__':
    freeze_support()  
    main()
