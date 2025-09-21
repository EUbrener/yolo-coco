# Execute utilizando:
# python detection.py --input diretorio_do_arquivo

import cv2
import time
import numpy as np
import argparse
import os

COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

with open("coco.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255)

def process_frame(frame):
    start = time.time()
    classes, scores, boxes = model.detect(frame, 0.1, 0.2)
    end = time.time()
    
    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[classid % len(COLORS)]
        label = f"{class_names[classid]}: {score:.2f}"
        cv2.rectangle(frame, box, color, 2)
        cv2.putText(frame, label, (box[0], box[1] -10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    fps_label = f"FPS: {round((1.0/(end - start)), 2)}"

    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    return frame

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detecção de objetos em imagem ou vídeo com YOLOv4-tiny")
    parser.add_argument("--input", type=str, required=True, help="Caminho para o arquivo de imagem ou vídeo")
    args = parser.parse_args()

    input_path = args.input
    _, file_extension = os.path.splitext(input_path)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    if file_extension.lower() in image_extensions:
        print(f"[INFO] Processando imagem: {input_path} ")
        image = cv2.imread(input_path)
        if image is None:
            print(f"[ERRO] Não foi possível ler a imagem: {input_path}")
        else:
            result_image = process_frame(image)
            cv2.imshow("Detecção em Imagem", result_image)
            cv2.waitKey(0)
    
    else:
        print(f"[INFO] Processando vídeo: {input_path} ")
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"[ERRO] Não foi possível abrir o vídeo: {input_path}")
        else:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[INFO] Fim do vídeo.")
                    break

                result_frame = process_frame(frame)
                cv2.imshow("Detecção em Vídeo", result_frame)

                if cv2.waitKey(1) == 27:
                    break

            cap.release()
    cv2.destroyAllWindows()

