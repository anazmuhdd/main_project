import RPi.GPIO as GPIO
import time

# --- Setup ---
GPIO.setmode(GPIO.BCM)
PIN = 26
GPIO.setup(PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
initial_model_path = 'obj1.pt'
secondary_model_path = 'best.pt'

print("Monitoring GPIO pin 16 (BCM mode)")

try:
    prev_state = GPIO.input(PIN)
    while True:
        state = GPIO.input(PIN)
        if state != prev_state:  # Detect change
            if state == GPIO.HIGH:
                print("🔴 Button Pressed – condition A triggered")
            else:
                print("🟢 Button Released – condition B triggered")
            prev_state = state
        time.sleep(0.05)  # Small debounce delay

except KeyboardInterrupt:
    print("Program stopped by user")

finally:
    GPIO.cleanup()
