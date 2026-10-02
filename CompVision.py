import cv2 as cv
from ultralytics import YOLO

# Carrega o modelo pré-treinado leve do YOLOv8
model = YOLO('yolov8n.pt')

# Conecta à webcam (índice 0)
cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao acessar a câmera.")
    exit()

print("Detector YOLO ativo!")
print("Controles de Filtro no teclado:")
print("  [r] -> Apenas Pessoas")
print("  [c] -> Apenas Veículos")
print("  [p] -> Apenas Plantas")
print("  [t] -> Mostrar Todos")
print("  [q] -> Sair do programa")

# Mapeamento de classes do dataset COCO
CLASSES_VEICULOS = [2, 3, 5, 7]
CLASSE_PESSOA = 0
CLASSE_PLANTA = 58

# Modo de filtro inicial ('todos', 'pessoa', 'veiculo', 'planta')
modo_filtro = 'todos'

while True:
    isTrue, frame = cap.read()
    if not isTrue:
        break

    # Roda a detecção nos quadros da webcam
    results = model(frame, verbose=False)[0]

    cont_pessoas = 0
    cont_veiculos = 0
    cont_plantas = 0

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        if conf > 0.40:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Detectar Pessoa (Verde) se o modo for 'todos' ou 'pessoa'
            if cls_id == CLASSE_PESSOA and modo_filtro in ['todos', 'pessoa']:
                cont_pessoas += 1
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv.putText(frame, f'Pessoa {conf:.2f}', (x1, y1 - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Detectar Veículo (Azul) se o modo for 'todos' ou 'veiculo'
            elif cls_id in CLASSES_VEICULOS and modo_filtro in ['todos', 'veiculo']:
                cont_veiculos += 1
                cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv.putText(frame, f'Veiculo {conf:.2f}', (x1, y1 - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Detectar Planta (Amarelo) se o modo for 'todos' ou 'planta'
            elif cls_id == CLASSE_PLANTA and modo_filtro in ['todos', 'planta']:
                cont_plantas += 1
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv.putText(frame, f'Planta {conf:.2f}', (x1, y1 - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Monta a informação do topo de acordo com o filtro ativo
    if modo_filtro == 'todos':
        info_texto = f"Filtro: TODOS | Pessoas: {cont_pessoas} | Veiculos: {cont_veiculos} | Plantas: {cont_plantas}"
    elif modo_filtro == 'pessoa':
        info_texto = f"Filtro: PESSOAS (r) | Total: {cont_pessoas}"
    elif modo_filtro == 'veiculo':
        info_texto = f"Filtro: VEICULOS (c) | Total: {cont_veiculos}"
    elif modo_filtro == 'planta':
        info_texto = f"Filtro: PLANTAS (p) | Total: {cont_plantas}"

    cv.putText(frame, info_texto, (20, 40),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv.imshow('Detector de Objetos (YOLOv8)', frame)

    # Leitura das teclas de atalho
    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        modo_filtro = 'pessoa'
    elif key == ord('c'):
        modo_filtro = 'veiculo'
    elif key == ord('p'):
        modo_filtro = 'planta'
    elif key == ord('t'):
        modo_filtro = 'todos'

cap.release()
cv.destroyAllWindows()