
import cv2
import dlib

def calculate_average_brightness(img, landmarks_list):
    if landmarks_list is None:
        return None

    x_values, y_values = zip(*landmarks_list)
    min_x, max_x = min(x_values), max(x_values)
    min_y, max_y = min(y_values), max(y_values)

    face_region = img[min_y:max_y, min_x:max_x]

    average_brightness = face_region.mean()

    return average_brightness

def classify_brightness(average_brightness):
    if average_brightness is None:
        return None

    brightness_map = {
        (0, 51): "Very Dark",
        (52, 102): "Dark",
        (103, 153): "Moderate",
        (154, 204): "Bright",
        (205, 255): "Very Bright"
    }

    for brightness_range, description in brightness_map.items():
        if brightness_range[0] <= average_brightness <= brightness_range[1]:
            return description

    return "Unknown"

# Load the face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_81_face_landmarks.dat")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detector(frame)

    if len(faces) > 0:
        shape = predictor(frame, faces[0])

        landmarks_list = [(shape.part(i).x, shape.part(i).y) for i in range(81)]

        # Calculate the average brightness of the face region
        average_brightness = calculate_average_brightness(frame, landmarks_list)

        # Classify brightness based on the average brightness
        brightness_category = classify_brightness(average_brightness)

        print(average_brightness, brightness_category)

        min_x, min_y, max_x, max_y = shape.rect.left(), shape.rect.top(), shape.rect.right(), shape.rect.bottom()
        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

        # Display the brightness category
        cv2.putText(frame, f"Brightness: {brightness_category}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Video Stream', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
