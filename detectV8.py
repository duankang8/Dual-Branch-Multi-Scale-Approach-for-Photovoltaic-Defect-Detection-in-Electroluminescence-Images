import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('E:/kan-guo-de-lun-wen/guang-fu-ban/ultralytics-main-8.24/ultralytics-main-7.25/runs/detect/train37/weights/best.pt') # select your model.pt path
    model.predict(source='E:/kan-guo-de-lun-wen/guang-fu-ban/ultralytics-main-8.24/ultralytics-main-7.25/ultralytics/test2',
                  imgsz=640,
                  project='runs/detect',
                  name='exp1',
                  save=True,
                  # conf=0.2,
                  # visualize=True # visualize model features maps
                )
