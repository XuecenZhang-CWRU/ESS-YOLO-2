from ultralytics import YOLO
import os

if __name__ == '__main__':
    # 加载模型
    model = YOLO(model="ultralytics/cfg/models/ESS-YOLO/efficient-mamba-T.yaml")  # 从头开始构建新模型
    # model = YOLO(model="ultralytics/cfg/models/ESS-YOLO/efficient-mamba-S.yaml")
    # model = YOLO(model="ultralytics/cfg/models/ESS-YOLO/efficient-mamba-B.yaml")
    # Use the model
    results = model.train(data="VOC.yaml", epochs=1, device='0', batch=8, seed=42,)  # 训练模型
