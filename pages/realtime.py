# pip install roboflow supervision opencv-python

import cv2
from roboflow import Roboflow
import supervision as sv
from supervision import Detections

# Fungsi untuk mengambil gambar dari kamera
def capture_image(filename="captured_image.jpg"):
    cap = cv2.VideoCapture(0)  # Menggunakan kamera default
    if not cap.isOpened():
        raise IOError("Tidak bisa membuka kamera")

    while True:
        ret, frame = cap.read()
        if not ret:
            raise IOError("Gagal menangkap gambar")

        cv2.imshow('Tekan Spasi untuk Memotret, Esc untuk Keluar', frame)
        
        key = cv2.waitKey(1)
        if key % 256 == 27:  # Tekan Esc untuk keluar
            print("Keluar tanpa memotret")
            break
        elif key % 256 == 32:  # Tekan Spasi untuk memotret
            cv2.imwrite(filename, frame)
            print(f"Gambar disimpan sebagai {filename}")
            break

    cap.release()
    cv2.destroyAllWindows()

# Memotret gambar
capture_image("captured_image.jpg")

# Inisialisasi Roboflow dengan API key
rf = Roboflow(api_key="BJUnFHOaWAc3UuoqFPlz")
project = rf.workspace("cv-e65se").project("obj_lemon")
version = project.version(1)

# Mengunduh model
model = version.model

# Prediksi menggunakan model
result = model.predict("captured_image.jpg", confidence=40, overlap=30).json()

# Ekstraksi label dari hasil prediksi
labels = [item["class"] for item in result["predictions"]]

# Konversi hasil prediksi menjadi format Detections dari supervision
detections = Detections.from_roboflow(result)

# Inisialisasi annotator
label_annotator = sv.LabelAnnotator()
box_annotator = sv.BoxAnnotator()

# Membaca gambar yang telah diambil
image = cv2.imread("captured_image.jpg")
if image is None:
    raise FileNotFoundError("Gambar tidak ditemukan atau tidak bisa dimuat.")

# Anotasi gambar dengan bounding box
annotated_image = box_annotator.annotate(
    scene=image, detections=detections)
# Anotasi gambar dengan label
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections, labels=labels)

# Menampilkan gambar yang telah dianotasi
sv.plot_image(image=annotated_image, size=(16, 16))
