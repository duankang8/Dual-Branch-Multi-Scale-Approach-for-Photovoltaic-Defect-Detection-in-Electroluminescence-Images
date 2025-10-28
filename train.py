import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/yolov8bifpn.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='ultralytics/cfg/datasets/NWPU VHR-10.yaml',
                cache=False,
                imgsz=640,
                epochs=150,
                batch=1,
                close_mosaic=0,
                workers=0,
                device='0',
                optimizer='SGD', # using SGD
                # patience=0, # close earlystop
                # resume='runs/train/exp10/weights/last.pt', # last.pt path
                amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )

