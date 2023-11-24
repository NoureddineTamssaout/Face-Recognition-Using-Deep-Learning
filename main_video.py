import cv2
from simple_facerec import SimpleFacerec

# Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

# Try different camera indices to find the correct one
for camera_index in range(10):
    # Load Camera
    cap = cv2.VideoCapture(camera_index)

    # Check if the camera is opened successfully
    if cap.isOpened():
        print(f"Camera index {camera_index} is working.")
        break
    else:
        print(f"Camera index {camera_index} is not working.")

# Check if no working camera index was found
if not cap.isOpened():
    print("Error: No working camera found.")
    exit()

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Check if the frame is valid
    if not ret or frame is None:
        print("Error: Failed to capture frame.")
        break

    # Print the frame shape for debugging
    print("Frame shape:", frame.shape)

    # Detect Faces
    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Check for the 'Esc' key to exit the loop
    key = cv2.waitKey(1)
    if key == 27:
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
