import lgpio
import time

# Define buzzer GPIO pin
BUZZER_PIN = 16

# Open GPIO chip
chip = lgpio.gpiochip_open(0)

def trigger_buzzer(duration=2):
    """Turn the buzzer on for the specified duration (in seconds)."""
    try:
        lgpio.gpio_claim_output(chip, BUZZER_PIN)
        lgpio.gpio_write(chip, BUZZER_PIN, 1)
        time.sleep(duration)
        lgpio.gpio_write(chip, BUZZER_PIN, 0)
    except Exception as e:
        print(f"Error triggering buzzer: {e}")

def cleanup():
    """Release GPIO resources."""
    lgpio.gpiochip_close(chip)

if __name__ == "__main__":
    try:
        print("Buzzer is ON for 2 seconds.")
        trigger_buzzer()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        cleanup()
