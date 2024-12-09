import cv2
from ultralytics import YOLO
import pyttsx3

# Các biến số:
# Chiều rộng thực của vật thể W(cm)
real_width = 15
# Khoảng cách thực từ vật thể đến camera D(cm)
real_distance = 68

# Nhận diện khuôn mặt
CascadeClass = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Khởi tạo mô hình YOLO và webcam
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

# Hàm tính tiêu cự
def focalLength(pixel_width, real_distance, real_width):
    return (pixel_width * real_distance) / real_width

# Hàm tính khoảng cách
def distanceCal(focal_length, real_width, width):
    return (focal_length * real_width) / width

# Hàm tính chiều rộng trên ảnh P(pixel)
def detectFace(img):
    width = 0
    BGRgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = CascadeClass.detectMultiScale(BGRgray)
    for face in faces:
        a,b,c,d = face
        cv2.rectangle(img,(a,b),(a+c,b+d),(255,255,0),3)
        width = c
    return width

# Ảnh tham khảo
img = cv2.imread('Ref_Photo.jpg')
# Tính P
pixel_width = detectFace(img)
# Tính tiêu cự F
focal_length = focalLength(pixel_width, real_distance, real_width)
print(focal_length)

# Khởi tạo pyttsx3 engine
engine = pyttsx3.init()

# Kiểm tra nếu không thể mở webcam
if not cap.isOpened():
    print("Error: Could not open webcam.")

# Từ điển ánh xạ các labels từ tiếng Anh sang tiếng Việt
labels_translation = {
    "person": "người",
    "bike": "xe đạp",
    "vehicle": "xe máy",
    "car": "xe hơi",
    "bus": "xe buýt",
    "truck": "xe công trình",
    "electric bike": "xe điện",
    "tree": "cái cây",
    "dog": "con chó",
    "cat": "con mèo",
    "stone chair": "ghế đá",
    "barrier": "hàng rào",
    "stairs": "cầu thang",
    "thresholds": "bậc thềm",
    "doors": "cánh cửa",
    "chair": "cái ghế",
    "dustbin": "thùng rác",
    "wall": "cái tường",
    "power poles": "cột điện",
    "pillar": "cây cột",
    "street lamp": "đèn đường",
    "sewage": "cái cống",
    "pothole" : "cái hố",
    "stone": "cục đá",
    # Thêm các labels khác nếu cần
}
def speak_label(label):
    # Chuyển đổi label từ tiếng Anh sang tiếng Việt
    vietnamese_label = labels_translation.get(label, label)
    return vietnamese_label

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

                # Tìm chiều rộng của bounding box và tính khoảng cách
                width = box.xywh[0][2]
                distance = distanceCal(focal_length, real_width, width)

                # Thêm thông tin vật thể và vị trí vào danh sách
                if distance <= 30:
                    detected_objects.add(f"Có một {speak_label(object_name)} ở {position} của bạn")
                    # detected_objects.add(f"There is a {object_name}")

            # Tìm vật thể mới chưa được đọc tên
            new_objects = detected_objects - last_detected_objects
            if new_objects:
                description = ', '.join(new_objects)
                last_detected_objects = detected_objects  # Cập nhật danh sách vật thể đã được đọc tên

                # Phát âm thanh từ chuỗi văn bản
                voices = engine.getProperty('voices')
                engine.setProperty('voice', voices[1].id)
                engine.setProperty('rate', 170)
                engine.say(description)
                engine.runAndWait()

    # Kiểm tra nếu phím 'q' được nhấn để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()