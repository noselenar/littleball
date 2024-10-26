from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('best.pt')  # 替换为你的模型路径

# 导出为 ONNX 格式
model.export(format='onnx')
