from ultralytics import YOLO
import torch
def train_model():
    # model = YOLO("E:/dk/ultralytics-main-2/ultralytics/cfg/models/v8/yolov8.yaml").load('yolov8n.pt')
    model = YOLO("E:/kan-guo-de-lun-wen/guang-fu-ban/ultralytics-main-8.24/ultralytics-main-7.25/runs/detect/train43/weights/best.pt")

    # E:/dk/ultralytics - main - 2/runs/detect/train3/weights/best.pt
    r = model.val(data="PV.yaml", epochs=300)
    print()

if __name__ == '__main__':
    train_model()
