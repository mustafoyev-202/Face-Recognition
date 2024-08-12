import threading
import cv2
from deepface import DeepFace


# Function to try different backends for video capture
def open_camera():
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_V4L2]
    for backend in backends:
        cap = cv2.VideoCapture(0, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if cap.isOpened():
            return cap
    return None


# Open camera with available backend
cap = open_camera()
if not cap:
    print("Error: Could not open video device.")
    exit()

# Initializations
counter = 0
face_match = False
reference_image = cv2.imread('Image.jpg')  # Change to your image path
if reference_image is None:
    print("Error: Could not read reference image.")
    cap.release()
    exit()

thread_lock = threading.Lock()


def check_face(frame):
    global face_match
    try:
        # Check if the face in the current frame matches the reference image
        result = DeepFace.verify(frame, reference_image, enforce_detection=False)
        with thread_lock:
            face_match = result['verified']
    except Exception as e:
        with thread_lock:
            face_match = False
        print("Error in face verification:", e)


while True:
    ret, frame = cap.read()

    if ret:
        # Process every 30th frame to reduce the load
        if counter % 30 == 0:
            threading.Thread(target=check_face, args=(frame.copy(),)).start()

        # Display result on the frame
        with thread_lock:
            display_text = 'MATCH' if face_match else 'NO MATCH'
            text_color = (0, 255, 0) if face_match else (255, 0, 0)
        cv2.putText(frame, display_text, (20, 450), cv2.FONT_HERSHEY_PLAIN, 2, text_color, 3)

        # Show the video frame
        cv2.imshow('Video', frame)
        counter += 1

    # Exit on 'q' key press
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
