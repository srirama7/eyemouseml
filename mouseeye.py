import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe Face Mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Get screen size
screen_w, screen_h = pyautogui.size()

# Initialize variables for mouse control
mouse_moved = False
prev_x, prev_y = 0, 0

# Start video capture
cam = cv2.VideoCapture(0)

while True:
    # Read frame from webcam
    ret, frame = cam.read()
    if not ret:
        break

    # Flip the frame horizontally for natural viewing
    frame = cv2.flip(frame, 1)

    # Convert BGR frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect face landmarks
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks

    # If landmarks are detected
    if landmark_points:
        landmarks = landmark_points[0].landmark

        # Extract landmarks for left eye
        left_eye_landmarks = [landmarks[159], landmarks[145]]

        # Calculate eye movement
        eye_y_diff = left_eye_landmarks[0].y - left_eye_landmarks[1].y

        # Threshold for eye blink detection
        blink_threshold = 0.004

        # Simulate mouse movement based on eye movement
        if eye_y_diff < blink_threshold:
            # Calculate screen coordinates
            screen_x = int(screen_w * left_eye_landmarks[0].x)
            screen_y = int(screen_h * left_eye_landmarks[0].y)

            # Move the mouse cursor
            pyautogui.moveTo(screen_x, screen_y)

            # Set mouse_moved flag
            mouse_moved = True
        else:
            # Reset mouse_moved flag if the user is not moving their eyes
            mouse_moved = False

        # Perform click action if the user blinked
        if mouse_moved and prev_x == screen_x and prev_y == screen_y:
            pyautogui.click()

        # Update previous mouse position
        prev_x, prev_y = screen_x, screen_y

        # Draw landmarks and eye circles on the frame for visualization
        for landmark in left_eye_landmarks:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

    # Display the frame
    cv2.imshow('Eye Controlled Mouse', frame)

    # Check for key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cam.release()
cv2.destroyAllWindows()
