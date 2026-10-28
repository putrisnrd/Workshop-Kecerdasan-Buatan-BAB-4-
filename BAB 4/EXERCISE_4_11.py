import cv2
import numpy as np
from keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import img_to_array

# Inisialisasi ukuran gambar dan model VGG19
image_size = 224
model = vgg19.VGG19(weights="imagenet")
print(model.summary())

# Membuka webcam (indeks 0)
camera = cv2.VideoCapture(0)

while camera.isOpened():
  ok, cam_frame = camera.read()
  if not ok:
    break

  # Pra-pemrosesan frame dari kamera untuk VGG19
  frame = cv2.resize(cam_frame, (image_size, image_size))
  numpy_image = img_to_array(frame)
  image_batch = np.expand_dims(numpy_image, axis=0)
  processed_image = vgg19.preprocess_input(image_batch.copy())

  # Mendapatkan prediksi probabilitas menggunakan VGG19
  predictions = model.predict(processed_image)
  label = decode_predictions(predictions)

  # Menampilkan hasil prediksi pada frame video
  text = f"VGG19: {label[0][0][1]}, {label[0][0][2]:.1f}"
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