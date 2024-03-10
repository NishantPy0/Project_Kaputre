import cv2
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os
import shutil
import time

# Constants
num_images_to_capture = 5
video_duration = 10  # seconds
temporary_folder = 'temporary_folder'

# Create temporary folder if not exists
if not os.path.exists(temporary_folder):
    os.makedirs(temporary_folder)

# Configure the email server and sender details
smtp_server = "smtp.gmail.com"
smtp_port = 587
sender_email = "sender_email"
sender_password = "sender_password"
receiver_email = "receiver_email"

# Video capture
cap = cv2.VideoCapture(0)  # Use the appropriate camera index or video file path

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(os.path.join(temporary_folder, 'recording.avi'), fourcc, 10, (640, 480))

# Motion detection loop
motion_detected = False
start_time = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for motion detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if start_time is not None and time.time() - start_time > video_duration:
        motion_detected = False
        start_time = None

    if not motion_detected:
        # Set the initial frame as a reference
        reference_frame = gray
        continue

    # Compute the absolute difference between the current frame and reference frame
    frame_delta = cv2.absdiff(reference_frame, gray)
    thresh = cv2.threshold(frame_delta, 30, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)


    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate over contours and check for motion
    for contour in contours:
        if cv2.contourArea(contour) > 1000:
            motion_detected = True
            start_time = time.time()

            # Capture images
            for i in range(num_images_to_capture):
                cv2.imwrite(os.path.join(temporary_folder, f'image_{i}.png'), frame)

            # Write frames to video
            for _ in range(int(video_duration) * 10):  # Assuming 10 frames per second
                video_writer.write(frame)

            break

    cv2.imshow("Motion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
video_writer.release()
cv2.destroyAllWindows()

# Send email with attachments
def send_email():
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = 'Motion Detected'

    attach_file(msg, os.path.join(temporary_folder, 'recording.avi'))
    for i in range(num_images_to_capture):
        attach_file(msg, os.path.join(temporary_folder, f'image_{i}.png'))

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())

def attach_file(msg, file_path):
    part = MIMEBase('application', 'octet-stream')
    part.set_payload(open(file_path, 'rb').read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', f'attachment; filename="{os.path.basename(file_path)}"')
    msg.attach(part)

send_email()

# Clean up temporary files
shutil.rmtree(temporary_folder)
