
import cv2
import dlib

# Load the face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_81_face_landmarks.dat")

cap = cv2.VideoCapture(0)

very_close_threshold = 0.423
close_threshold = 0.260
moderate_threshold = 0.130
far_threshold = 0.065

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detector(frame)

    if len(faces) > 0:
        shape = predictor(frame, faces[0])

        landmarks_list = [(shape.part(i).x, shape.part(i).y) for i in range(81)]
        x_values, y_values = zip(*landmarks_list)
        min_x, max_x = min(x_values), max(x_values)
        min_y, max_y = min(y_values), max(y_values)

        face_area = (max_x - min_x) * (max_y - min_y)
        
        frame_height, frame_width, _ = frame.shape
        face_area_ratio = face_area / (frame_height * frame_width)

        # Classify the distance based on the current face area
        if face_area_ratio > very_close_threshold:
            distance_category = "Very Close"
        elif very_close_threshold >= face_area_ratio > close_threshold:
            distance_category = "Close"
        elif close_threshold >= face_area_ratio > moderate_threshold:
            distance_category = "Moderate"
        elif moderate_threshold >= face_area_ratio > far_threshold:
            distance_category = "Far"
        else:
            distance_category = "Very Far"

        print(face_area_ratio, distance_category)

        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

        # Display the distance category
        cv2.putText(frame, f"Distance: {distance_category}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Video Stream', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()