import cv2
from imutils import face_utils
import dlib
import numpy as np
import pyttsx3
import time
import os
os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"

font = cv2.FONT_HERSHEY_SIMPLEX

# Function to save the images
def save_image(image, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, filename), image)

# Function to resize the images to (64x64)
def resize_image(image, width=640, height=640):
    return cv2.resize(image, (width, height))

name_classes = ["Drowsy", "Neutral"]

# Load the TensorFlow Lite model using the TFLite runtime
from tflite_runtime.interpreter import Interpreter

interpreter = Interpreter(model_path="model_cnn.tflite")
interpreter.allocate_tensors()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# face landmark
predictor_path = "shape_predictor_68_face_landmarks.dat" 
predictor = dlib.shape_predictor(predictor_path)

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Set up the video capture
cap = cv2.VideoCapture(0) # Raspberry Pi Camera V2

frame_interval = 5  # Interval in seconds to process each frame.
last_frame_time = time.time()

# Modified
TIME_INT = 2*60
start_time = time.time()
drowsy_count = 0
dc = 0
nc = 0
neu_count = 0
engine = pyttsx3.init()
# Modified
log_file = open("detection_log.txt", "a")  # Open the log file in append mode

last_predicted_array = []
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Check if the 'q' key is pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)


    current_time = time.time()
    if current_time - last_frame_time >= frame_interval:
        # Reset counts for each frame
        dc = 0
        nc= 0
        
        for (x, y, w, h) in faces:
            # Crop Face
            crop_img = resize_image(gray[y:y+h, x:x+w])

            # Landmark
            h , w = crop_img.shape
            face_rect = dlib.rectangle(0, 0, 0 + w, 0 + h)
            landmarks = predictor(crop_img, face_rect)
            landmarks = face_utils.shape_to_np(landmarks)

            # Crop Landmark Based
            feature_indices = list(range(18, 68))
            feature_points = np.array([(landmarks[i][0], landmarks[i][1]) for i in feature_indices])

            for (x_feat, y_feat) in feature_points:
                cv2.circle(crop_img, (x_feat, y_feat), 2, (0, 255, 0), -1)

            x_feat, y_feat, w_feat, h_feat = cv2.boundingRect(feature_points)
            crop_img = crop_img[y_feat:y_feat + h_feat, x_feat:x_feat + w_feat]
            crop_img = resize_image(crop_img, 64, 64)

            image_tocnn = cv2.resize(crop_img, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
            image_tocnn = np.expand_dims(np.array(image_tocnn, dtype=np.float32), axis=0)
            image_tocnn /= 255.0  # Normalize to [0, 1]
            image_tocnn = image_tocnn.reshape(input_details[0]['shape'])
            
            interpreter.set_tensor(input_details[0]['index'], image_tocnn)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            predicted_class = np.argmax(output_data[0])
            last_predicted_array.append(name_classes[predicted_class])
            
            # Count Drowsy and Neutral for each frame
            if name_classes[predicted_class] == "Drowsy":
                drowsy_count += 1
            if name_classes[predicted_class] == "Neutral":
                neu_count += 1
                
            if name_classes[predicted_class] == "Drowsy":
                dc += 1
            if name_classes[predicted_class] == "Neutral":
                nc += 1
            
        last_frame_time = current_time

    # Write detections to the log file
    for (x, y, w, h), predicted in zip(faces, last_predicted_array):
        current_time_str = time.strftime('%H:%M:%S')
        log_line = f"{current_time_str}: Detected: {dc} Drowsy, {nc} Neutral, {round((current_time - last_frame_time) * 1000, 1)}ms\n"
        log_file.write(log_line)
    
    # Modified
    elapsed_time = current_time - start_time
    if elapsed_time >= TIME_INT:
        ann = f"Students detected Drowsy {drowsy_count} times in the past 15 minutes."
        print(ann)
        
        engine.say(ann)
        engine.runAndWait()
        
        start_time = current_time
        drowsy_count = 0
        neu_count = 0
    
    index = 0
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if index < len(last_predicted_array):
            cv2.putText(frame, last_predicted_array[index], (x, y - 10), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            index = index + 1

    cv2.imshow("Output", frame)

# Close the log file, release the video capture, and close all windows
log_file.close()
cap.release()
cv2.destroyAllWindows()
