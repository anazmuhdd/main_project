import cv2
import time
from ultralytics import YOLO
from picamera2 import Picamera2
import RPi.GPIO as GPIO
import pyttsx3
import pygame

# -------------------- GPIO Setup --------------------
GPIO.setmode(GPIO.BCM)
BUTTON_PIN = 17   # GPIO pin for mode switch button
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# -------------------- TTS Setup --------------------
engine = pyttsx3.init()
engine.setProperty('rate', 130)

def speak(text):
    """Speak text using pyttsx3 safely."""
    engine.say(text)
    engine.runAndWait()

# -------------------- YOLO Models --------------------
currency_model = YOLO("./best.pt")     # Your trained currency detection model
object_model = YOLO("/home/viewsense/viewsense/obj1.pt")     # Your general object detection model
currency_classes = currency_model.names
object_classes = object_model.names

# -------------------- Camera Setup --------------------
picam2 = Picamera2()
picam2.preview_configuration.main.size = (1280, 720)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()
time.sleep(0.1)

# -------------------- Audio Setup --------------------
pygame.init()
pygame.mixer.init()

# -------------------- Mode Control --------------------
mode = "currency"  # default mode

def switch_mode(channel):
    """Toggle between currency and object recognition modes."""
    global mode
    mode = "object" if mode == "currency" else "currency"
    speak(f"Switched to {mode} detection mode")
    print(f"Mode changed to: {mode}")

# Detect button press to toggle mode
GPIO.add_event_detect(BUTTON_PIN, GPIO.FALLING, callback=switch_mode, bouncetime=1000)

# -------------------- Main Loop --------------------
threshold = 0.5
print("Starting detection... Press Ctrl+C to stop.")
speak("Starting currency detection mode")

try:
    while True:
        frame = picam2.capture_array()

        # Choose model based on mode
        if mode == "currency":
            results = currency_model(frame)[0]
            class_list = currency_classes
        else:
            results = object_model(frame)[0]
            class_list = object_classes

        for box in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = box
            if score >= threshold:
                label = class_list[int(class_id)]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {score:.2f}", (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Speak the detected class
                speak(f"Detected {label}")

                # Optional GPIO feedback or audio feedback
                if label == '20':
                    if not pygame.mixer.music.get_busy():
                        pygame.mixer.music.load("20.mp3")
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy():
                            time.sleep(0.1)

                elif label == '100':
                    if not pygame.mixer.music.get_busy():
                        pygame.mixer.music.load("100.mp3")
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy():
                            time.sleep(0.1)

        # Show the frame (optional)
        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nDetection stopped by user.")
    speak("Detection stopped")

finally:
    cv2.destroyAllWindows()
    picam2.close()
    GPIO.cleanup()
    pygame.mixer.quit()
    pygame.quit()
