import os
from ultralytics import YOLO

def main():
    # 데이터셋 경로 설정
    dataset_path = os.path.join(os.getcwd(), 'dataset')
    data_yaml = os.path.join(dataset_path, 'data.yaml')

    # YOLO 모델 훈련
    model = YOLO('yolov8s.pt')
    model.train(data=data_yaml, epochs=100, imgsz=800, plots=True)

if __name__ == '__main__':
    main()
