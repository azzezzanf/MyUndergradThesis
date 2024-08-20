import cv2
from imutils import face_utils
import dlib

import numpy as np
from tflite_runtime.interpreter import Interpreter

import time
import os
os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"

font = cv2.FONT_HERSHEY_SIMPLEX
name_classes = ["Drowsy", "Neutral"]
interpreter = Interpreter(model_path="model_cnn.tflite")
interpreter.allocate_tensors()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
predictor_path = "shape_predictor_68_face_landmarks.dat" 
predictor = dlib.shape_predictor(predictor_path)

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

cap = cv2.VideoCapture(0)
frame_interval = 5  # Interval in seconds to process each frame.
last_frame_time = time.time()
last_predict_emotion = ""

while True:
    _, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    if len(faces) == 1:
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, last_predict_emotion, (x, y - 10), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        current_time = time.time()
        if current_time - last_frame_time >= frame_interval:
            x, y, w, h = faces[0]
            crop_img = gray[y:y+h, x:x+w]
            crop_img = cv2.resize(crop_img, (640, 640))

            facial_landmarks = []
            for (x, y, w, h) in faces:
                h , w = crop_img.shape
                face_rect = dlib.rectangle(0, 0, 0 + w, 0 + h)
                landmarks = predictor(crop_img, face_rect)
                landmarks = face_utils.shape_to_np(landmarks)
                facial_landmarks.append(landmarks)

                feature_indices = list(range(18, 68))
                feature_points = np.array([(landmarks[i][0], landmarks[i][1]) for i in feature_indices])
            
            x, y, w, h = cv2.boundingRect(feature_points)
            crop_img = crop_img[y:y + h, x:x + w]
            crop_img = cv2.resize(crop_img, (64, 64))

            image_tocnn = cv2.resize(crop_img, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
            image_tocnn = np.expand_dims(np.array(image_tocnn, dtype=np.float32), axis=0)
            image_tocnn /= 255.0  # Normalize to [0, 1]
            image_tocnn = image_tocnn.reshape(input_details[0]['shape'])

            interpreter.set_tensor(input_details[0]['index'], image_tocnn)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            predicted_class = np.argmax(output_data[0])

            last_predict_emotion = name_classes[predicted_class]
            print("Predicted class:", name_classes[predicted_class])
            last_frame_time = current_time
    
    
    cv2.imshow("Output", image)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()

