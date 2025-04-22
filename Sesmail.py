import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email(sender_email, sender_password, recipient_email, subject, body):
    # Create the MIME object
    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = recipient_email
    message['Subject'] = subject

    # Attach the email body
    message.attach(MIMEText(body, 'plain'))

    try:
        # Set up the SMTP server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()  # Enable TLS
        server.login(sender_email, sender_password)

        # Send the email
        server.sendmail(sender_email, recipient_email, message.as_string())
        print("Email sent successfully!")

    except Exception as e:
        print(f"Error sending email: {e}")

    finally:
        server.quit()

# Example usage
if __name__ == "__main__":
    sender_email = "nsanzestevo1@gmail.com"  # Replace with your Gmail address
    sender_password = "prib ogff rytd mtya"  # Replace with your App Password
    recipient_email = "gasasiras103@gmail.com"  # Replace with recipient's email
    subject = "Test Email from Python"
    body = "This is a test email sent from a Python script!"

    send_email(sender_email, sender_password, recipient_email, subject, body)