# Example 4.15
#https://pypi.org/project/image-classifiers/
import cv2
from classification_models.keras import Classifiers
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
import numpy as np

clf, preprocess_input = Classifiers.get("vgg16")
sz = 224

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

    text = f"{label[0][0][1]}, {label[0][0][2]:.1f}"
    cv2.putText(
        cam_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
    )
    cv2.imshow("Classification", cam_frame)

    key = cv2.waitKey(30)
    if key == 27:
        break

camera.release()
cv2.destroyAllWindows()