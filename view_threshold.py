import cv2
import numpy as np

# Initialize camera
cap = cv2.VideoCapture(0)

# Set the initial frame as None
previous_frame = None

# Set the initial background color as black
background_color = (0, 0, 0)


alpha = 0.05 # (0.5 chito     0.05 dhilo  )

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve motion detection
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if previous_frame is None:
        # Set the initial frame for the first iteration
        previous_frame = gray.copy().astype(float)
    else:
        # Update the background using running average
        cv2.accumulateWeighted(gray, previous_frame, alpha)
        background = cv2.convertScaleAbs(previous_frame)

        # Calculate the absolute difference between the current frame and the background
        frame_diff = cv2.absdiff(background, gray)

        

        _, thresh = cv2.threshold(frame_diff, 60, 1, cv2.THRESH_BINARY)

        # Update the frame to be completely black
        frame[:, :] = 0

        # Set the color to white for the regions with motion
        frame[thresh != 0] =225

    # Display the result
    cv2.imshow("Motion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
