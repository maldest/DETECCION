import time
import cv2
from ultralytics import YOLO

# --- Configuración ---
AVAILABLE_CLASSES = ['organic', 'inorganic']   # ajusta si tu modelo usa otros nombres
IMGSZ = 640
CONF  = 0.40          # ajusta según tu escena
LOG_EVERY = 2.0       # segundos entre mensajes en consola

print("Clases a identificar disponibles:", AVAILABLE_CLASSES)
selected = input("Selecciona una clase objetivo ('organic' o 'inorganic'): ").strip().lower()
while selected not in AVAILABLE_CLASSES:
    selected = input("Selecciona 'organic' o 'inorganic': ").strip().lower()
opposite = 'inorganic' if selected == 'organic' else 'organic'
print(f"[INFO] Clase objetivo seleccionada: {selected}")

# --- Cámara ---
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # usa CAP_MSMF si te va mejor
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# --- Modelo YOLO ---
model = YOLO('best.pt')

def get_model_names(m):
    try:
        if hasattr(m, 'names') and isinstance(m.names, dict):
            return {int(k): str(v).lower() for k, v in m.names.items()}
    except Exception:
        pass
    return {0:'organic', 1:'inorganic'}

names = get_model_names(model)
print("[INFO] Clases del modelo:", names)

last_log = 0.0

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[ERROR] No se pudo capturar frame.")
            break

        # Inferencia silenciosa
        res = model(frame, imgsz=IMGSZ, conf=CONF, verbose=False)[0]

        # Flags de presencia por clase
        allowed_present = False
        forbidden_present = False

        if res.boxes is not None and len(res.boxes) > 0:
            cls_ids = res.boxes.cls.int().tolist()
            confs   = res.boxes.conf.tolist()
            for cid, c in zip(cls_ids, confs):
                cname = names.get(cid, 'unknown').lower()
                if c >= CONF:
                    if cname == selected:
                        allowed_present = True
                    if cname == opposite:
                        forbidden_present = True

        # Mensaje cada LOG_EVERY segundos
        now = time.time()
        if now - last_log >= LOG_EVERY:
            if forbidden_present:
                # >>> Mensaje de alerta SIN el texto entre paréntesis <<<
                print(f"[ALERTA] Objeto {opposite} detectado donde no corresponde.")
            else:
                print(f"[OK] Objeto {selected} detectado: todo correcto.")
            last_log = now

        # Vista anotada (opcional)
        annotated = res.plot()
        cv2.imshow("Detección de residuos", annotated)

        # Salir con ESC
        if cv2.waitKey(1) == 27:
            print("[INFO] Saliendo (ESC).")
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
