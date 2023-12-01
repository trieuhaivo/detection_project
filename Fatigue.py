import cv2
import dlib
import numpy as np
import time

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Load the pre-trained face and eye detector models from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

# Threshold for eye closure detection
eye_closure_threshold = 0.3

# Tracking eye closure duration
eyes_closed_start_time = None
eyes_closed_duration_threshold = 5  # Duration threshold in seconds

while True:
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        landmarks_list = [(p.x, p.y) for p in landmarks.parts()]

        # Extract the left and right eye coordinates
        left_eye_start, left_eye_end = 42, 48
        right_eye_start, right_eye_end = 36, 42
        left_eye_points = np.array(landmarks_list[left_eye_start:left_eye_end], np.int32)
        right_eye_points = np.array(landmarks_list[right_eye_start:right_eye_end], np.int32)

        # Calculate the aspect ratio for both eyes
        left_eye_aspect_ratio = eye_aspect_ratio(left_eye_points)
        right_eye_aspect_ratio = eye_aspect_ratio(right_eye_points)

        # Check if eyes are closed
        if left_eye_aspect_ratio < eye_closure_threshold and right_eye_aspect_ratio < eye_closure_threshold:
            if eyes_closed_start_time is None:
                eyes_closed_start_time = time.time()
            else:
                eyes_closed_duration = time.time() - eyes_closed_start_time

                if eyes_closed_duration > eyes_closed_duration_threshold:
                    cv2.putText(frame, "Fatigue Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            eyes_closed_start_time = None

        cv2.polylines(frame, [left_eye_points], isClosed=True, color=(255, 255, 139), thickness=1)
        cv2.polylines(frame, [right_eye_points], isClosed=True, color=(255, 255, 139), thickness=1)

    cv2.imshow("Eye Tracking for Fatigue Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()