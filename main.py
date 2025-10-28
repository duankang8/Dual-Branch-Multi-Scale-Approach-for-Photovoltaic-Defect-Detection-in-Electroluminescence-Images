from ultralytics import YOLO
import torch
def train_model():
    model = YOLO("E:/kan-guo-de-lun-wen/guang-fu-ban/ultralytics-main-8.24/ultralytics-main-7.25/yolov8nbifpn.yaml").load('yolov8n.pt')
    model.train(data="PV.yaml", epochs=300)

if __name__ == '__main__':
    train_model()




