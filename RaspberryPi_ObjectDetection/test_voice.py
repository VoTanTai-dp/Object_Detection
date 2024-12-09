import cv2
from ultralytics import YOLO
import pyttsx3

# Khởi tạo mô hình YOLO và webcam
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

# Khởi tạo pyttsx3 engine
engine = pyttsx3.init()

# Kiểm tra nếu không thể mở webcam
if not cap.isOpened():
    print("Error: Could not open webcam.")

# Biến để theo dõi vật thể đã được đọc tên
last_detected_objects = set()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    results = model.predict(source=frame, save=False, conf=0.2, iou=0.5)

    if len(results) > 0:
        result = results[0]
        if len(result.boxes) > 0:
            plot = result.plot()
            cv2.imshow("Webcam Detection", plot)

            # Lấy tên và vị trí của các vật thể được phát hiện
            detected_objects = set()
            for box in result.boxes:
                # Lấy tên vật thể
                object_name = model.names[int(box.cls[0])]

                # Tính toán vị trí của bounding box
                x_center = (box.xywh[0][0] + box.xywh[0][2] / 2) / frame.shape[1]

                # Xác định vị trí trái, giữa, phải
                if x_center < 1 / 3:
                    position = "bên trái"
                elif x_center < 2 / 3:
                    position = "trước mặt"
                else:
                    position = "bên phải"

                # Thêm thông tin vật thể và vị trí vào danh sách
                detected_objects.add(f"có một {object_name} ở {position} của bạn")

            # Tìm vật thể mới chưa được đọc tên
            new_objects = detected_objects - last_detected_objects
            if new_objects:
                description = ', '.join(new_objects)
                last_detected_objects = detected_objects  # Cập nhật danh sách vật thể đã được đọc tên

                # Phát âm thanh từ chuỗi văn bản
                voices = engine.getProperty('voices')
                engine.setProperty('voice', voices[1].id)
                engine.setProperty('rate', 150)
                engine.say(description)
                engine.runAndWait()

    # Kiểm tra nếu phím 'q' được nhấn để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
