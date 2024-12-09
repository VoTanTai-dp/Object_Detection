import pyttsx3

# Khởi tạo engine
engine = pyttsx3.init()
# Từ điển ánh xạ các labels từ tiếng Anh sang tiếng Việt
labels_translation = {
    "person": "người",
    "car": "xe hơi",
    "bicycle": "xe đạp",
    "dog": "chó",
    "cat": "mèo",
    # Thêm các labels khác nếu cần
}

# Liệt kê tất cả các giọng đọc có sẵn
voices = engine.getProperty('voices')
for voice in voices:
    if 'vi_VN' in voice.languages:
        engine.setProperty('voice', voice.id)
        break

# Đặt tốc độ đọc (tùy chỉnh theo ý muốn)
engine.setProperty('rate', 150)  # Tốc độ đọc có thể điều chỉnh theo nhu cầu

# Hàm đọc labels bằng tiếng Việt
def speak_label(label):
    # Chuyển đổi label từ tiếng Anh sang tiếng Việt
    vietnamese_label = labels_translation.get(label, label)  # Nếu không tìm thấy, sử dụng label gốc
    # Đọc label bằng tiếng Việt
    engine.setProperty('voice', voices[1].id)
    engine.say(vietnamese_label)
    engine.runAndWait()

# Ví dụ: đọc các labels
labels = ["person", "car", "bicycle"]
for label in labels:
    speak_label(label)
