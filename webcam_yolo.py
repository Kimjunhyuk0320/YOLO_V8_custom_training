import cv2
from ultralytics import YOLO
import tkinter as tk
from tkinter import messagebox

# YOLO 모델 로드
local_model_path = 'runs/detect/train/weights/best.pt'
model = YOLO(local_model_path)

# 클래스 이름 목록 (모델에 맞게 수정 필요)
class_names = model.names

# 웹캠 열기
cap = cv2.VideoCapture(0)  # 0은 기본 웹캠을 의미합니다

# 웹캠이 열리지 않으면 종료
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    
    # 프레임 읽기에 실패하면 종료
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break
    
    # YOLO v8로 객체 탐지, conf=0.5 설정
    results = model(frame, conf=0.5)
    
    # 탐지된 객체 표시
    annotated_frame = results[0].plot()
    
    # 확률이 0.70 이상인 객체 이름 표시
    for result in results[0].boxes:
        if result.conf > 0.70:
            class_id = int(result.cls)
            label = class_names[class_id]
            
            # 텍스트 크기 계산
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = frame.shape[0] - 20  # 화면 아래에서 20px 위
            
            # 화면 중앙 아래에 텍스트 표시
            cv2.putText(annotated_frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # 결과 프레임 보여주기
    cv2.imshow('YOLO v8 Webcam', annotated_frame)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
