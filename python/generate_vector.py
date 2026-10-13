import cv2
import numpy as np

# --- CÁC THÔNG SỐ CẦN THAY ĐỔI ---
image_path = "reference.jpg"
detection_model_path = "face_detection_yunet.onnx"
recognition_model_path = "face_recognition_sface_2021dec.onnx"
output_file_path = "demo_001_embedding.bin"  # THAY ĐỔI Ở ĐÂY
# ------------------------------------

print("Đang tải các mô hình...")
try:
    # Tải mô hình phát hiện khuôn mặt (YuNet)
    face_detector = cv2.FaceDetectorYN.create(
        detection_model_path,
        "",
        (320, 320),  # kích thước đầu vào mặc định
        score_threshold=0.6,  # tăng độ nhạy
        nms_threshold=0.3,
        top_k=5000
    )

    # Tải mô hình nhận dạng khuôn mặt (SFace)
    face_recognizer = cv2.FaceRecognizerSF.create(
        recognition_model_path,
        ""
    )
except Exception as e:
    print(f"Lỗi khi tải mô hình: {e}")
    exit()

print("Tải mô hình thành công.")

print(f"Đang đọc ảnh từ: {image_path}")
image = cv2.imread(image_path)
if image is None:
    print("Lỗi: Không thể đọc được file ảnh. Vui lòng kiểm tra lại đường dẫn.")
    exit()

# Đảm bảo ảnh ở dạng BGR 3 kênh
if image.shape[2] == 4:
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

# Resize ảnh về kích thước hợp lý (YuNet cần ảnh ít nhất 320x320)
h, w = image.shape[:2]
if w < 320 or h < 320:
    image = cv2.resize(image, (max(320, w), max(320, h)))

# Cập nhật lại input size cho bộ phát hiện
face_detector.setInputSize((image.shape[1], image.shape[0]))

print("Đang phát hiện khuôn mặt trong ảnh...")
retval, faces = face_detector.detect(image)

if faces is None or len(faces) == 0:
    print("Không tìm thấy khuôn mặt nào trong ảnh.")
    exit()

print(f"Tìm thấy {len(faces)} khuôn mặt.")

# Chọn khuôn mặt đầu tiên (thường là lớn nhất)
face = faces[0]

print("Đang căn chỉnh và trích xuất vector đặc trưng...")
aligned_face = face_recognizer.alignCrop(image, face)
feature = face_recognizer.feature(aligned_face)

print("Trích xuất thành công. Đang lưu vector ra file...")
try:
    # CÁCH 1: Sử dụng NumPy (khuyến nghị)
    feature_array = np.array(feature)
    feature_array.tofile(output_file_path)
    
    print(f"Đã lưu thành công vector đặc trưng vào file: '{output_file_path}'")
    print(f"Kích thước vector: {feature_array.shape}")
except Exception as e:
    print(f"Lỗi khi lưu file: {e}")

# (Tuỳ chọn) Hiển thị vùng khuôn mặt đã phát hiện
for det in faces:
    box = list(map(int, det[:4]))
    cv2.rectangle(image, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0,255,0), 2)
cv2.imshow("Detected Face", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

