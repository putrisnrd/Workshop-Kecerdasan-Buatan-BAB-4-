import cv2
from classification_models.keras import Classifiers
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
import numpy as np

# Pilih model yang ingin digunakan: 'vgg16', 'resnet50', 'mobilenetv2', 'densenet201', atau 'inceptionv3'
selected_model = "resnet50"

if selected_model == "vgg16":
    clf, preprocess_input = Classifiers.get("vgg16")
    sz = 224
elif selected_model == "resnet50":
    clf, preprocess_input = Classifiers.get("resnet50")
    sz = 224
elif selected_model == "mobilenetv2":
    clf, preprocess_input = Classifiers.get("mobilenetv2")
    sz = 224
elif selected_model == "densenet201":
    clf, preprocess_input = Classifiers.get("densenet201")
    sz = 224
elif selected_model == "inceptionv3":
    clf, preprocess_input = Classifiers.get("inceptionv3")
    sz = 299
else:
    raise ValueError("Model tidak dikenal!")

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

    text = f"{selected_model.upper()}: {label[0][0][1]}, {label[0][0][2]:.1f}"
    cv2.putText(
        cam_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
    )
    cv2.imshow("Classification", cam_frame)

    key = cv2.waitKey(30)
    if key == 27:
        break

camera.release()
cv2.destroyAllWindows()