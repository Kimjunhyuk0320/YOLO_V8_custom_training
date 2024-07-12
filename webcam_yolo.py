import cv2
from ultralytics import YOLO

# YOLO 모델 로드
# model = YOLO('yolov8n.pt')

# 상대 경로에서 YOLO 모델 로드
local_model_path = 'runs/detect/train/weights/best.pt'
model = YOLO(local_model_path)

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
    
    # YOLO v8로 객체 탐지
    results = model(frame)
    
    # 탐지된 객체 표시
    annotated_frame = results[0].plot()
    
    # 결과 프레임 보여주기
    cv2.imshow('YOLO v8 Webcam', annotated_frame)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
