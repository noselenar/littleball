import time

from ultralytics import YOLO

model = YOLO('yolo11m.pt')
results = model.train(data='ultralytics/cfg/datasets/Mydata.yaml', epochs=5, imgsz=[1920, 1200], device=[0, 1], workers=0, batch=8 , cache=True)
time.sleep(10)
