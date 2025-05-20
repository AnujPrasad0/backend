import cv2
import mediapipe as mp
import pyautogui
import math
import random
import util  # Assumes util provides get_angle() and get_distance()
from pynput.mouse import Button, Controller

print("Virtual Mouse Script is Running...")


# For volume control using pycaw
from ctypes import POINTER, cast
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# ------------------------
# SETUP FOR VOLUME CONTROL
# ------------------------

# Initialize volume interface using pycaw
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume_interface = cast(interface, POINTER(IAudioEndpointVolume))

def set_system_volume(volume_scalar):
    """
    Sets the system master volume.
    volume_scalar should be between 0.0 (mute) and 1.0 (max volume).
    """
    volume_scalar = max(0.0, min(1.0, volume_scalar))
    volume_interface.SetMasterVolumeLevelScalar(volume_scalar, None)

def process_volume_control(hand_landmarks, frame):
    """
    Controls volume using the distance between the thumb and index finger on the left hand.
    Maps the distance between landmarks 4 and 8 to a 0.0-1.0 range and updates system volume.
    """
    thumb = hand_landmarks.landmark[4]
    index = hand_landmarks.landmark[8]
    x1, y1 = int(thumb.x * frame.shape[1]), int(thumb.y * frame.shape[0])
    x2, y2 = int(index.x * frame.shape[1]), int(index.y * frame.shape[0])
    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Calculate Euclidean distance (pixels)
    distance = math.hypot(x2 - x1, y2 - y1)
    min_dist = 30    # distance corresponding to volume off
    max_dist = 150   # distance corresponding to max volume

    distance = max(min_dist, min(max_dist, distance))
    volume_ratio = (distance - min_dist) / (max_dist - min_dist)
    set_system_volume(volume_ratio)

    volume_value = int(volume_ratio * 100)
    cv2.putText(frame, f"Volume: {volume_value}%", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame

# ------------------------
# SETUP FOR MOUSE CONTROL (RIGHT HAND)
# ------------------------

# Initialize the mouse controller (using pynput)
mouse = Controller()

# Global variables for mouse pointer smoothing.
previous_x, previous_y = None, None
smoothing_factor = 0.30  # Adjust as needed

# Initialize MediaPipe Hands (for both hands)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Get screen dimensions for mapping normalized coordinates.
screen_width, screen_height = pyautogui.size()

def recognize_gesture(hand_landmarks, frame):
    """
    Uses selected landmarks from the right hand to determine if a gesture corresponds 
    to a click or scrolling.
    """
    thumb_tip   = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip   = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip  = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    index_dip   = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
    index_pip   = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_dip  = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
    middle_pip  = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    pinky_tip   = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_dip   = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP]
    pinky_pip   = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
    ring_tip    = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_dip    = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP]
    ring_pip    = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]

    if isScrolling(index_tip.x, index_tip.y, index_dip.x, index_dip.y, index_pip.y,
                   middle_tip.x, middle_tip.y, middle_dip.x, middle_dip.y, middle_pip.y,
                   pinky_tip.y, pinky_dip.y, pinky_pip.y, ring_tip.y, ring_dip.y, ring_pip.y):
        return "Scrolling"
    if isLeftClicking(thumb_tip.y, index_tip.y):
        return "Left Click"
    if isRightClicking(thumb_tip.y, middle_tip.y):
        return "Right Click"
    return "N/A"

def isLeftClicking(thumb_tip_y, index_tip_y):
    return abs(thumb_tip_y - index_tip_y) < 0.05

def isRightClicking(thumb_tip_y, middle_tip_y):
    return abs(thumb_tip_y - middle_tip_y) < 0.05

def isScrolling(index_tip_x, index_tip_y,
                index_dip_x, index_dip_y, index_pip_y,
                middle_tip_x, middle_tip_y,
                middle_dip_x, middle_dip_y, middle_pip_y,
                pinky_tip_y, pinky_dip_y, pinky_pip_y,
                ring_tip_y, ring_dip_y, ring_pip_y):
    diff_tip = abs(index_tip_x - middle_tip_x)
    diff_dip = abs(index_dip_x - middle_dip_x)
    isPinkyLower = (pinky_tip_y > pinky_pip_y and pinky_dip_y > pinky_pip_y)
    isRingLower = (ring_tip_y > ring_pip_y)
    isIndexAbove = (index_tip_y < index_pip_y and index_dip_y < index_pip_y)
    isMiddleAbove = (middle_tip_y < middle_pip_y and middle_dip_y < middle_pip_y)
    
    return diff_tip < 0.05 and diff_dip < 0.05 and isPinkyLower and isRingLower and isIndexAbove and isMiddleAbove

def process_mouse_control(hand_landmarks, frame):
    """
    Processes right-hand landmarks to control the mouse pointer movement.
    It calculates the pointer position from an average of specific landmarks,
    applies smoothing, moves the pointer, and then checks for and executes
    click/scroll gestures.
    """
    global previous_x, previous_y, smoothing_factor

    # Calculate pointer coordinates using an average of:
    # - INDEX_FINGER_MCP
    # - RING_FINGER_MCP
    # - WRIST (to get a more stable average)
    scaling_factor = 6.0
    index_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    ring_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    
    index_tip_x = (index_finger_mcp.x + ring_finger_mcp.x) / 2
    index_tip_y = (ring_finger_mcp.y + wrist.y) / 2

    x = int((index_tip_x * screen_width - (screen_width * 0.5)) * scaling_factor) - screen_width
    y = int((index_tip_y * screen_height - (screen_height * 0.7)) * scaling_factor)
    
    if previous_x is not None and previous_y is not None:
        x = int(previous_x * (1 - smoothing_factor) + x * smoothing_factor)
        y = int(previous_y * (1 - smoothing_factor) + y * smoothing_factor)
    
    try:
        pyautogui.moveTo(x, y)
    except Exception as e:
        print("Error moving the mouse:", e)
    
    previous_x, previous_y = x, y

    # Determine the gesture based on hand landmarks.
    gesture = recognize_gesture(hand_landmarks, frame)
    print("Detected Gesture:", gesture)  # Debug print

    try:
        if gesture == "Left Click":
            mouse.press(Button.left)
            mouse.release(Button.left)
        elif gesture == "Right Click":
            mouse.press(Button.right)
            mouse.release(Button.right)
        elif gesture == "Scrolling":
            pyautogui.scroll(-10)
    except Exception as e:
        print("Error executing mouse action:", e)
    
    cv2.putText(frame, gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (200, 200, 200), 2, cv2.LINE_AA)
    return frame

# ------------------------
# MAIN LOOP: INTEGRATED VOLUME & MOUSE CONTROL
# ------------------------

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame for a mirror view and convert color for MediaPipe.
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed = hands.process(rgb_frame)
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        if processed.multi_hand_landmarks and processed.multi_handedness:
            for hand_landmarks, handedness in zip(processed.multi_hand_landmarks,
                                                  processed.multi_handedness):
                label = handedness.classification[0].label  # "Left" or "Right"
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                if label == "Left":
                    frame = process_volume_control(hand_landmarks, frame)
                elif label == "Right":
                    frame = process_mouse_control(hand_landmarks, frame)

        cv2.imshow("Integrated Gesture Control", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
