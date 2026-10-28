# Example 4.14
import cv2
import numpy as np
from keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications import (
    inception_v3,
    mobilenet,
    resnet50,
    vgg16,
)
from tensorflow.keras.preprocessing.image import img_to_array

# Inisialisasi model (saat ini menggunakan ResNet50)
# model = vgg16.VGG16(weights='imagenet')
model = resnet50.ResNet50(weights="imagenet")
# model = mobilenet.MobileNet(weights='imagenet')
# model = inception_v3.InceptionV3(weights='imagenet')

print(model.summary())

# Membuka webcam (indeks 0)
camera = cv2.VideoCapture(0)
image_size = 224
# image_size = 299  # Digunakan jika menggunakan InceptionV3

while camera.isOpened():
  ok, cam_frame = camera.read()
  if not ok:
    break

  frame = cv2.resize(cam_frame, (image_size, image_size))
  numpy_image = img_to_array(frame)
  image_batch = np.expand_dims(numpy_image, axis=0)

  # Pra-pemrosesan gambar sesuai model yang digunakan
  # processed_image = vgg16.preprocess_input(image_batch.copy())
  processed_image = resnet50.preprocess_input(image_batch.copy())
  # processed_image = mobilenet.preprocess_input(image_batch.copy())
  # processed_image = inception_v3.preprocess_input(image_batch.copy())

  # Mendapatkan prediksi probabilitas untuk setiap kelas
  predictions = model.predict(processed_image)
  label = decode_predictions(predictions)

  # Menampilkan hasil prediksi pada frame video
  text = f"{label[0][0][1]}, {label[0][0][2]:.1f}"
  cv2.putText(
      cam_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2
  )
  cv2.imshow("video image", cam_frame)

  # Tekan tombol 'ESC' (ASCII: 27) untuk keluar dari loop
  key = cv2.waitKey(30)
  if key == 27:
    break

# Melepaskan kamera dan menutup semua jendela OpenCV
camera.release()
cv2.destroyAllWindows()