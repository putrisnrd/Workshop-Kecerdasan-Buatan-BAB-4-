import cv2
from classification_models.keras import Classifiers
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
import numpy as np

# Daftar model dan ukuran input yang tersedia pada library image-classifiers
# Ubah indeks tuple untuk menguji model yang berbeda:
# 0: ('vgg16', 224)
# 1: ('resnet50', 224)
# 2: ('mobilenetv2', 224)
# 3: ('densenet201', 224)
# 4: ('inceptionv3', 299)

models_config = [
    ("vgg16", 224),
    ("resnet50", 224),
    ("mobilenetv2", 224),
    ("densenet201", 224),
    ("inceptionv3", 299),
]

# Pilih model berdasarkan indeks (Contoh: Indeks 1 untuk ResNet50)
model_name, sz = models_config[1]

clf, preprocess_input = Classifiers.get(model_name)
model = clf(input_shape=(sz, sz, 3), weights="imagenet", classes=1000)
print(model.summary())

camera = cv2.VideoCapture(0)

while camera.isOpened():
  ret, cam_frame = camera.read()
  if not ret:
    break

  frame = cv2.resize(cam_frame, (sz, sz))
  image = np.asarray(frame, dtype=np.float32)
  image = np.expand_dims(image, axis=0)
  image = preprocess_input(image)

  preds = model.predict(image)
  label = decode_predictions(preds)

  text = f"{model_name.upper()}: {label[0][0][1]}, {label[0][0][2]:.1f}"
  cv2.putText(
      cam_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
  )
  cv2.imshow("Classification Comparison", cam_frame)

  key = cv2.waitKey(30)
  if key == 27:
    break

camera.release()
cv2.destroyAllWindows()