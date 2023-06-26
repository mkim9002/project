from ultralytics import YOLO
import torch
import os
import multiprocessing

def main():
    # print(torch.cuda.is_available())
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    model = YOLO('yolov8n.pt')
    model.train(data='./face_aihub.v1i.yolov8/data.yaml', epochs=10000, patience=40, batch= 48, seed= 1030)
    model.predict(source='C:/project/yolo8/face_aihub.v1i.yolov8/test/images/', save=True)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()