# Import thư viện YOLO từ ultralytics và thư viện OpenCV
from ultralytics import YOLO
import cv2


# Định nghĩa hàm display_results với đầu vào là kết quả detection
def display_results(result):
    # Kiểm tra nếu có ít nhất một bounding box trong kết quả
    if len(result.boxes) > 0:
        # Lặp qua từng bounding box
        for box in result.boxes:
            # Lấy tên loại đối tượng (class_id) từ kết quả detection
            class_id = result.names[box.cls[0].item()]
            # Lấy tọa độ của bounding box và chuyển đổi chúng thành danh sách các số nguyên
            cords = box.xyxy[0].tolist()
            cords = [round(x) for x in cords]
            # Lấy xác suất (độ tin cậy) của detection và làm tròn đến 2 chữ số thập phân
            conf = round(box.conf[0].item(), 2)
            # In ra loại đối tượng
            print("Object type:", class_id)
            # In ra tọa độ của bounding box
            print("Coordinates:", cords)
            # In ra xác suất của detection
            print("Probability:", conf)
            # In dấu gạch ngang để phân biệt các đối tượng
            print("---")

        # Vẽ bounding boxes lên hình ảnh gốc và lưu trữ hình ảnh đã vẽ
        plot = result.plot()
        # Hiển thị hình ảnh với các bounding box qua cửa sổ OpenCV
        cv2.imshow("Detection Result", plot)
        # Đợi người dùng nhấn phím bất kỳ để đóng cửa sổ
        cv2.waitKey(0)
        # Đóng tất cả các cửa sổ OpenCV
        cv2.destroyAllWindows()
    else:
        # In ra thông báo nếu không có đối tượng nào được phát hiện trong ảnh
        print("No detections found in the image.")


# Định nghĩa hàm process_image với đầu vào là mô hình và đường dẫn ảnh
def process_image(model, image_path):
    # Dự đoán kết quả trên ảnh, lưu ảnh và thiết lập ngưỡng độ tin cậy và iou
    results = model.predict(source=image_path, save=True, conf=0.2, iou=0.5)
    # Kiểm tra nếu có ít nhất một kết quả dự đoán
    if len(results) > 0:
        # Lấy kết quả dự đoán đầu tiên
        result = results[0]
        # Gọi hàm display_results để hiển thị kết quả
        display_results(result)
    else:
        # In ra thông báo nếu không có kết quả nào được trả về từ mô hình
        print("No results returned by the model.")


# Định nghĩa hàm process_webcam với đầu vào là mô hình
def process_webcam(model):
    # Mở webcam
    #url = "http://192.168.1.3:4747/video"
    #cap = cv2.VideoCapture(url)
    cap = cv2.VideoCapture(0)
    # Kiểm tra nếu không thể mở webcam
    if not cap.isOpened():
        # In ra thông báo lỗi nếu không thể mở webcam
        print("Error: Could not open webcam.")
        return
    # Vòng lặp liên tục để đọc frame từ webcam
    while True:
        # Đọc frame từ webcam
        ret, frame = cap.read()
        # Kiểm tra nếu không thể đọc frame
        if not ret:
            # In ra thông báo lỗi nếu không thể chụp ảnh
            print("Error: Failed to capture image.")
            break
        # Dự đoán kết quả trên frame từ webcam, không lưu ảnh và thiết lập ngưỡng độ tin cậy và iou
        results = model.predict(source=frame, save=False, conf=0.2, iou=0.5)
        # Kiểm tra nếu có ít nhất một kết quả dự đoán
        if len(results) > 0:
            # Lấy kết quả dự đoán đầu tiên
            result = results[0]
            # Kiểm tra nếu có ít nhất một bounding box trong kết quả
            if len(result.boxes) > 0:
                # Vẽ bounding boxes lên frame và lưu trữ hình ảnh đã vẽ
                plot = result.plot()
                # Hiển thị hình ảnh với bounding boxes qua cửa sổ OpenCV
                cv2.imshow("Webcam Detection", plot)
        # Kiểm tra nếu phím 'q' được nhấn để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Giải phóng webcam
    cap.release()
    # Đóng tất cả các cửa sổ OpenCV
    cv2.destroyAllWindows()

def main():
    # Tải mô hình YOLO
    model = YOLO("yolov8n.pt")
    # Yêu cầu người dùng chọn chế độ (hình ảnh hoặc webcam)
    mode = input("Select mode (image/webcam): ").strip().lower()

    # Kiểm tra chế độ được chọn là 'image'
    if mode == 'image':
        # Lấy đường dẫn ảnh từ người dùng
        # image_path = input("Enter image path: ")
        image_path = "coco128/coco128/images/train2017/Hr_32.jpg"
        # Gọi hàm process_image để xử lý ảnh
        process_image(model, image_path)
    # Kiểm tra chế độ được chọn là 'webcam'
    elif mode == 'webcam':
        # Gọi hàm process_webcam để xử lý webcam
        process_webcam(model)
    else:
        # In ra thông báo nếu chế độ được chọn không hợp lệ
        print("Invalid mode selected. Choose 'image' or 'webcam'.")


# Kiểm tra nếu script được chạy trực tiếp
if __name__ == "_main_":
    # Gọi hàm main để bắt đầu chương trình
    main()