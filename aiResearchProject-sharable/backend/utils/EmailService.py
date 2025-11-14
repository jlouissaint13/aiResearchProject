import os
from dotenv import load_dotenv
load_dotenv()
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText



def send_email(recipient, subject, body):
    sender_email = os.getenv("sender_email")
    sender_password = os.getenv("sender_password")
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()


            server.login(sender_email, sender_password)
            server.send_message(msg)
    except Exception as e:
        print("Email error:", e)


class EmailService:
    pass