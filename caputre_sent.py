import cv2
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import os
import shutil
import time
import winsound  # Import winsound for beep sound on Windows

# Setting email
sender_email = "sender_emai@gmail.com"
app_pass = "passcode / app code"
receiver_email = "reciever's email"

# motion threshold
motion_threshold = 10000

#  the temporary folder
temp_folder = "temp_images"

# Create the temporary folder if it doesn't exist
if not os.path.exists(temp_folder):
    os.makedirs(temp_folder)

# Set up the email server
server = smtplib.SMTP("smtp.gmail.com", 587)
server.starttls()
server.login(sender_email, app_pass)

# Initialize video capture
cap = cv2.VideoCapture(0)

# Allow some time for the camera to stabilize
time.sleep(2)

_, prev_frame = cap.read()

while True:
    _, frame = cap.read()

    # Calculate absolute difference between current and previous frames
    diff = cv2.absdiff(prev_frame, frame)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion_detected = False

    for contour in contours:
        if cv2.contourArea(contour) > motion_threshold:
            motion_detected = True
            break

    cv2.imshow("Camera Feed", frame)

    if motion_detected:
        # Play beep sound
        winsound.Beep(1000, 200)  # Frequency: 1000 Hz, Duration: 200 ms

        # Save the image
        image_path = os.path.join(temp_folder, "motion_image.jpg")
        cv2.imwrite(image_path, frame)

        # Send email with the image
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = 'Motion Detected'

        body = 'Motion detected! Check the attached image.'
        msg.attach(MIMEText(body, 'plain'))

        with open(image_path, 'rb') as img_file:
            img = MIMEImage(img_file.read())
            img.add_header('Content-Disposition', 'attachment', filename="motion_image.jpg")
            msg.attach(img)

        server.sendmail(sender_email, receiver_email, msg.as_string())

        # Clear the temporary folder
        shutil.rmtree(temp_folder)
        os.makedirs(temp_folder)

    prev_frame = frame

    # Exit the loop and close the window when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
server.quit()
