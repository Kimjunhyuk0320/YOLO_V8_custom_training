import cv2
import os
from ultralytics import YOLO

# YOLO 모델 로드
local_model_path = 'runs/detect/train/weights/best.pt'
model = YOLO(local_model_path)

# 테스트 이미지 폴더 경로
test_img_folder = 'testImg'

# 결과 저장 폴더 경로
output_folder = 'output'
os.makedirs(output_folder, exist_ok=True)

# 폴더 내 모든 .jpg 파일에 대해 객체 탐지 수행
for img_file in os.listdir(test_img_folder):
    if img_file.endswith('.jpg'):
        # 이미지 경로
        img_path = os.path.join(test_img_folder, img_file)

        # 이미지 읽기
        frame = cv2.imread(img_path)
        
        if frame is None:
            print(f"이미지를 읽을 수 없습니다: {img_path}")
            continue
        
        # YOLO v8로 객체 탐지
        results = model(frame)
        
        # 탐지된 객체 표시
        annotated_frame = results[0].plot()
        
        # 결과 이미지 저장 경로
        output_path = os.path.join(output_folder, img_file)
        
        # 결과 이미지 저장
        cv2.imwrite(output_path, annotated_frame)
        print(f"결과 저장됨: {output_path}")

print("모든 이미지에 대한 객체 탐지 완료.")
