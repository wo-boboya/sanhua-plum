import warnings, os

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # 设置环境变量禁用wandb
    os.environ['WANDB_DISABLED'] = 'true'
    model = YOLO(r'/home/xbs/三华李/ultralytics_6360/SimAM_1.yaml')
    model.train(
        data=r'/home/xbs/三华李/ultralytics_6360/sanhau_6360.yaml',
        imgsz=416,
        epochs=200,
        batch=32,
        workers=4,
        name='SimAM_1',
        device='0',
        save=True,
        amp=True
    )


# from ultralytics.models import YOLO
# import os
#
# model=YOLO("/home/xbs/三华李/ultralytics-yolo11-main_0827/runs/detect/sanhua_all_v12/weights/last.pt")
# results=model.train(resume=True)