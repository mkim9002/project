from ultralytics import YOLO
import zipfile
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

model = YOLO('yolov8n.pt')

if __name__ == '__main__':

    model.predict(source='C:/project/yolo8/face_aihub.v1i.yolov8/test/images/', save=True)
    





