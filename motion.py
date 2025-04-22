import RPi.GPIO as GPIO
import time

# Pin Definitions
relay_pin = 26        # GPIO pin for relay (light control)
motion_pin = 17       # GPIO pin for PIR motion sensor
ldr_do_pin = 4        # GPIO pin for LDR sensor

# GPIO Setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(relay_pin, GPIO.OUT)
GPIO.setup(motion_pin, GPIO.IN)
GPIO.setup(ldr_do_pin, GPIO.IN)

# Ensure relay is OFF at start
#GPIO.output(relay_pin, GPIO.HIGH)

print("System initialized. Monitoring motion and light...")

try:
    while True:
        motion_detected = GPIO.input(motion_pin)  # 1 = Motion detected, 0 = No motion
        light_status = GPIO.input(ldr_do_pin)    # 0 = Dark, 1 = Bright

        # AND logic: Turn ON light if both motion is detected and it's dark
        if motion_detected == 1 and light_status == 0:
          #  print("Motion detected and it's dark - Relay ON (Light ON)")
            GPIO.output(relay_pin, GPIO.LOW)  # Turn ON light
        else:
          #  print("No motion or too bright - Relay OFF (Light OFF)")
            GPIO.output(relay_pin, GPIO.HIGH)   # Turn OFF light

        time.sleep(0.5)

except KeyboardInterrupt:
    print("Exiting program...")

finally:
    GPIO.output(relay_pin, GPIO.LOW)  # Reset relay
    GPIO.cleanup()

