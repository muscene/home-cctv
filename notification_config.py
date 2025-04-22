# Notification configuration
NOTIFICATION_CONFIG = {
    'email': {
        'enabled': True,
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'username': 'your-email@gmail.com',
        'password': 'your-app-password',
        'recipients': ['recipient1@example.com', 'recipient2@example.com'],
        'cooldown_period': 300  # Seconds between notifications
    },
    'sms': {
        'enabled': False,
        'service': 'twilio',
        'account_sid': 'your-twilio-sid',
        'auth_token': 'your-twilio-token',
        'from_number': '+1234567890',
        'to_numbers': ['+1987654321'],
        'cooldown_period': 600  # Seconds between notifications
    },
    'push': {
        'enabled': False,
        'service': 'pushover',
        'api_token': 'your-pushover-token',
        'user_key': 'your-pushover-user-key',
        'cooldown_period': 300  # Seconds between notifications
    }
}