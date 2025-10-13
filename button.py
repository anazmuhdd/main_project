from gpiozero import Button, LED
from signal import pause

# --- Pin setup ---
BUTTON_PIN = 26  # GPIO pin where your button is connected
LED_PIN = 27      # GPIO pin for LED (or any output)

# Create objects
button = Button(BUTTON_PIN, pull_up=True)  # Button with pull-up resistor
led = LED(LED_PIN)

# --- Define actions ---
def button_pressed():
    print("Button pressed – LED ON")
    led.on()

def button_released():
    print("Button released – LED OFF")
    led.off()

# Assign actions
button.when_pressed = button_pressed
button.when_released = button_released

# Keep the program running
pause()
