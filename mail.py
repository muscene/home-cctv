import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders
import os

def send_email(sender_email, sender_password, smtp_server, smtp_port, recipient_email, subject, message_body, image_path=None, video_path=None):
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject

    # Attach the message body
    msg.attach(MIMEText(message_body, 'plain'))

    # Attach the image if provided
    if image_path and os.path.exists(image_path):
        try:
            with open(image_path, 'rb') as img_file:
                img = MIMEImage(img_file.read(), name=os.path.basename(image_path))
                msg.attach(img)
        except Exception as e:
            print(f"Error attaching image: {e}")

    # Attach the video if provided
    if video_path and os.path.exists(video_path):
        try:
            with open(video_path, 'rb') as vid_file:
                mime_base = MIMEBase('application', 'octet-stream')
                mime_base.set_payload(vid_file.read())
                encoders.encode_base64(mime_base)
                mime_base.add_header('Content-Disposition', f'attachment; filename="{os.path.basename(video_path)}"')
                msg.attach(mime_base)
        except Exception as e:
            print(f"Error attaching video: {e}")

    try:
        # Connect to SMTP server with SSL
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())
            print("Email sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")

# Example Usage
send_email(
    sender_email="iot@vrt.rw",
    sender_password="TheGreat@123",                # Replace with the actual password
    smtp_server="mail.vrt.rw",                     # Correct SMTP server for your domain
    smtp_port=465,                                 # Use 465 for SSL
    recipient_email="virtualrwandatechnology@gmail.com",
    subject="Home Security Alert",
    message_body="Mr. Steve, here is the security alert with image and video.",
    image_path="unknown_faces/unknown_20241210_113515.jpg",                # Replace with the actual path to the image
    video_path="unknown_faces/video.mp4"                 # Replace with the actual path to the 3-second video
)
