markdown
# YOLO_V8_Custom_Training

## 캐릭터 이미지 라벨링 및 학습

YOLOv8을 사용하여 객체 탐지를 구현하는 방법을 안내합니다.

## 환경 세팅

### 1. Python 설치
먼저, Python을 설치합니다. [Python 공식 웹사이트](https://www.python.org/downloads/)에서 최신 버전을 다운로드하여 설치하세요.

### 2. CUDA 설치
NVIDIA GPU를 사용하여 학습 속도를 높이려면 CUDA를 설치해야 합니다. [CUDA 다운로드 페이지](https://developer.nvidia.com/cuda-downloads)에서 GPU에 맞는 버전을 설치하세요.

### 3. PyTorch 설치
PyTorch를 설치합니다. 터미널에서 Python 설치 폴더로 이동한 후, 아래 명령어를 실행하세요:
```sh
pip install torch torchvision torchaudio
YOLOv8 설치
YOLOv8을 설치하려면 터미널에서 아래 명령어를 실행하세요:

sh
pip install ultralytics

데이터 라벨링
Roboflow에서 데이터를 라벨링하고 YOLOv8 형식으로 dataset.zip 파일을 내보냅니다. Roboflow를 사용하여 데이터셋을 라벨링합니다.

YOLOv8 커스텀 데이터셋으로 학습시키기
train_yolo.py 스크립트를 사용하여 커스텀 데이터셋으로 YOLOv8을 학습시킵니다. 터미널에서 아래 명령어를 실행하세요:

sh

python train_yolo.py
train_yolo.py 파일 내용은 다음과 같습니다:

python

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
YOLOv8으로 새 이미지 탐지하기
img_yolo.py 스크립트를 사용하여 새로운 이미지를 탐지합니다. 터미널에서 아래 명령어를 실행하세요:

sh

python img_yolo.py
img_yolo.py 파일 내용은 다음과 같습니다:

python

import cv2
from ultralytics import YOLO
import os

# YOLO 모델 로드
local_model_path = 'runs/detect/train/weights/best.pt'
model = YOLO(local_model_path)

# 테스트 이미지 폴더 경로
test_img_folder = 'testImg'
output_folder = 'output'

# 출력 폴더가 없으면 생성
os.makedirs(output_folder, exist_ok=True)

# 테스트 이미지 폴더 내의 모든 이미지 파일 처리
for img_file in os.listdir(test_img_folder):
    if img_file.endswith('.jpg'):
        img_path = os.path.join(test_img_folder, img_file)
        img = cv2.imread(img_path)

        # YOLO v8로 객체 탐지
        results = model(img)

        # 탐지된 객체 표시
        annotated_img = results[0].plot()

        # 결과 이미지 저장
        output_path = os.path.join(output_folder, img_file)
        cv2.imwrite(output_path, annotated_img)

print("탐지가 완료되었습니다. 결과는 'output' 폴더를 확인하세요.")
탐지 결과는 output 폴더에 저장됩니다.

프로젝트 파일 구조
kotlin

YOLO_V8_Custom_Training/
│
├── dataset/
│   ├── images/
│   ├── labels/
│   └── data.yaml
│
├── runs/
│   └── detect/
│       └── train/
│           └── weights/
│               └── best.pt
│
├── testImg/
│   ├── image1.jpg
│   └── image2.jpg
│
├── output/
│   ├── image1.jpg
│   └── image2.jpg
│
├── train_yolo.py
└── img_yolo.py
go


위의 내용을 `README.md` 파일로 저장하시면 됩니다. 이 파일은 프로젝트의 개요, 환경 설정, 데이터 라벨링