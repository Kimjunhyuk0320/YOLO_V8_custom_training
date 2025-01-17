# YOLO_V8_Custom_Training with Roboflow
![YOLOv8 Logo](https://yolov8.org/wp-content/uploads/2024/01/YOLOv8.png) 
![Roboflow Logo1](https://cdn.prod.website-files.com/5f6bc60e665f54545a1e52a5/642746dba53a59a614a64b35_roboflow-open-graph.png) 

## 캐릭터 이미지 라벨링 및 학습

YOLOv8을 사용하여 객체 탐지를 구현하는 방법을 안내합니다.

## 환경 세팅

### 1. Python 설치
먼저, Python을 설치합니다. [Python 공식 웹사이트](https://www.python.org/downloads/)에서 최신 버전을 다운로드하여 설치하세요.

### 2. CUDA 설치
NVIDIA GPU를 사용하여 학습 속도를 높이려면 CUDA를 설치해야 합니다. [CUDA 다운로드 페이지](https://developer.nvidia.com/cuda-downloads)에서 GPU에 맞는 버전을 설치하세요.

### 3. PyTorch 설치
PyTorch를 설치합니다. 터미널에서 Python 설치 폴더로 이동한 후, 아래 명령어를 실행하세요:
[PyTorch 다운로드 설명 참조](https://wandb.ai/onlineinference/YOLO/reports/YOLOv5-Object-Detection-on-Windows-Step-By-Step-Tutorial---VmlldzoxMDQwNzk4)
```pip3 install ~~~```

### YOLOv8 설치
YOLOv8을 설치하려면 터미널에서 아래 명령어를 실행하세요:
```pip install ultralytics```

### 데이터 라벨링
Roboflow에서 데이터를 라벨링하고 YOLOv8 형식으로 dataset.zip 파일을 내보냅니다. Roboflow를 사용하여 데이터셋을 라벨링합니다. (이미 이 단계는 완료해서 git프로젝트 내에 첨부 완료)

### YOLOv8 커스텀 데이터셋으로 학습시키기
train_yolo.py 스크립트를 사용하여 커스텀 데이터셋으로 YOLOv8을 학습시킵니다. 터미널에서 아래 명령어를 실행하세요:

```python train_yolo.py```

### YOLOv8으로 새 이미지 탐지하기
img_yolo.py 스크립트를 사용하여 새로운 이미지를 탐지합니다. 터미널에서 아래 명령어를 실행하세요:

```python img_yolo.py```

탐지 결과는 output 폴더에 저장됩니다.

