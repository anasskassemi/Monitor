import cv2
import mediapipe as mp
import pyautogui

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Screen dimensions for scaling hand movements to screen size
screen_width, screen_height = pyautogui.size()

# Function to detect if a finger is raised
def is_finger_up(landmarks, finger_tip_id, finger_dip_id):
    return landmarks[finger_tip_id].y < landmarks[finger_dip_id].y

# Linear interpolation (LERP) function for smooth motion
def lerp(start, end, t):
    return start + (end - start) * t

# Initialize variables for mouse smoothing
mouse_x, mouse_y = pyautogui.position()
smooth_factor = 0.2  # Adjust for smoothness

# Initialize MediaPipe Hands
with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty frame.")
            continue

        # Flip the image horizontally for a selfie view
        image = cv2.flip(image, 1)

        # Convert the image color (OpenCV uses BGR, MediaPipe uses RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image to find hands
        hand_results = hands.process(image_rgb)

        # Draw hand landmarks on the image if any hand is detected
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Draw landmarks and connections
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get index finger tip coordinates
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_x = int(index_finger_tip.x * screen_width)
                index_y = int(index_finger_tip.y * screen_height)

                # Smooth mouse movement
                mouse_x = lerp(mouse_x, index_x, smooth_factor)
                mouse_y = lerp(mouse_y, index_y, smooth_factor)
                pyautogui.moveTo(int(mouse_x), int(mouse_y))

                # Detect if specific fingers are raised
                index_up = is_finger_up(hand_landmarks.landmark, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_DIP)
                middle_up = is_finger_up(hand_landmarks.landmark, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_DIP)
                ring_up = is_finger_up(hand_landmarks.landmark, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_DIP)

                fingers_up = [index_up, middle_up, ring_up]
                num_fingers_up = sum(fingers_up)

                # Perform action based on fingers up
                if fingers_up == [True, False, False]:  # Only index finger up
                    pyautogui.click()

                elif num_fingers_up == 2:
                    pyautogui.scroll(10)

                elif num_fingers_up == 3:
                    pyautogui.rightClick()
                elif num_fingers_up == 1:
                    pyautogui.leftClick()

        cv2.imshow('Hand Gesture Control', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
