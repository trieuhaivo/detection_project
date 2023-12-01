
import cv2
import dlib
from deepface import DeepFace
import numpy as np

# Load the pre-trained models
age_model = DeepFace.build_model(model_name="Age")

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load dlib's face detector and shape predictor for 81 landmarks
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_81_face_landmarks.dat")

video_capture = cv2.VideoCapture(0)

skip_frames = 5
frame_count = 0

while True:
    ret, frame = video_capture.read()

    frame_count += 1
    if frame_count % skip_frames != 0:
        continue

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector(gray_frame)

    for face in faces:
        # Extract the face ROI (Region of Interest)
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        # Resize the face ROI to match the input shape of the age model
        resized_face = cv2.resize(frame[y:y + h, x:x + w], (224, 224), interpolation=cv2.INTER_AREA)

        normalized_face = resized_face / 255.0

        # Reshape the image to match the input shape of the age model
        reshaped_face = normalized_face.reshape(1, 224, 224, 3)

        age_predictions = age_model.predict(reshaped_face)[0]
        predicted_age = np.argmax(age_predictions)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"Age: {predicted_age}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Real-time Age Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()