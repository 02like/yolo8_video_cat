"""
将 YOLOv8n 导出为 OpenVINO IR 格式（只需运行一次）
"""
from ultralytics import YOLO

# 下载并导出 YOLOv8 nano 模型
model = YOLO("yolov8n.pt")
model.export(format="openvino", imgsz=640, half=False)
print("导出完成: yolov8n_openvino_model/")
