
import cv2
import dlib
from deepface import DeepFace
import time

# Load the pre-trained emotion detection model
emotion_model = DeepFace.build_model(model_name = "Emotion")

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_81_face_landmarks.dat")

video_capture = cv2.VideoCapture(0)

skip_frames = 5
frame_count = 0

while True:
    ret, frame = video_capture.read()

    # Skip frames
    frame_count += 1
    if frame_count % skip_frames != 0:
        continue

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector(gray_frame)

    for face in faces:
        # Extract the face ROI
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        landmarks = landmark_predictor(gray_frame, face)

        for i in range(81):
            cv2.circle(frame, (landmarks.part(i).x, landmarks.part(i).y), 1, (255, 255, 255), -1)

        # Resize the face ROI to match the input shape of the emotion model
        resized_face = cv2.resize(gray_frame[y:y + h, x:x + w], (48, 48), interpolation=cv2.INTER_AREA)

        normalized_face = resized_face / 255.0

        # Reshape the image to match the input shape of the emotion model
        reshaped_face = normalized_face.reshape(1, 48, 48, 1)

        emotion_predictions = emotion_model.predict(reshaped_face)[0]
        predicted_emotion_index = emotion_predictions.argmax()
        predicted_emotion = emotion_labels[predicted_emotion_index]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Real-time Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()