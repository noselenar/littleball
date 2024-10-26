from ultralytics import YOLO


import os
import sys
sys.path.insert(0, os.path.abspath('/kaggle/working/littleball/trainsball'))


# 确定使用的 GPU 设备
device = 'cuda:0,1'  # 使用两个 GPU
#import sys
#sys.path.append('/kaggle/working/ultralytics')
# 加载模型
model = YOLO('yolo11m.pt')

# 训练模型
results = model.train(
    data='ultralytics/cfg/datasets/Mydata.yaml',  # 数据集路径
    epochs=5,
    imgsz=[1920, 1200],
    device=device,  # 指定使用的设备
    workers=0,      # 使用的工作进程数量
    batch=8,        # 每个 GPU 的批量大小
    cache=True
)
