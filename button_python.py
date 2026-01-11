import cv2
import time
from ultralytics import YOLO
from picamera2 import Picamera2
from gpiozero import Button
import pyttsx3
import pygame
import threading

# -------------------- Button Setup --------------------
BUTTON_PIN = 26  # GPIO pin connected to your button
button = Button(BUTTON_PIN, pull_up=True)

engine = pyttsx3.init()
engine.setProperty('rate', 130)

# Lock to ensure thread-safe speaking
speak_lock = threading.Lock()

def speak(text):
    """Speak text safely from a separate thread."""
    def _speak():
        with speak_lock:
            engine.say(text)
            engine.runAndWait()
    threading.Thread(target=_speak, daemon=True).start()

# -------------------- YOLO Models --------------------
currency_model = YOLO("./best.pt")      # Currency detection model
object_model = YOLO("./obj1.pt")        # Object detection model
currency_classes = currency_model.names
object_classes = object_model.names

# -------------------- Camera Setup --------------------
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()
time.sleep(0.1)

# -------------------- Audio Setup --------------------
pygame.init()
pygame.mixer.init()

# -------------------- Mode Control --------------------
mode = "currency"  # Default mode

def switch_mode():
    """Toggle between currency and object recognition modes."""
    global mode
    mode = "object" if mode == "currency" else "currency"
    speak(f"Switched to {mode} detection mode")
    print(f"Mode changed to: {mode}")

# Bind button press to mode switch
button.when_pressed = switch_mode

# -------------------- Main Detection Loop --------------------
threshold = 0.5
print("Starting detection... Press Ctrl+C to stop.")
speak("Starting currency detection mode")

try:
    while True:
        frame = picam2.capture_array()

        # Select model based on current mode
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
                cv2.putText(frame, f"{label} {score:.2f}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

         

                # Play audio feedback for certain currency notes
                if label == '20':
                    if not pygame.mixer.music.get_busy():
                        pygame.mixer.music.load("20.mp3")
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy():
                            time.sleep(0.1)

                elif label == '50':
                    if not pygame.mixer.music.get_busy():
                        pygame.mixer.music.load("50.mp3")
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy():
                            time.sleep(0.1)

                elif label == '100':
                    if not pygame.mixer.music.get_busy():
                        pygame.mixer.music.load("100.mp3")
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy():
                            time.sleep(0.1)

                elif label == '500':
                    if not pygame.mixer.music.get_busy():
                        pygame.mixer.music.load("500.mp3")
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy():
                            time.sleep(0.1)

                elif label == '10':
                    if not pygame.mixer.music.get_busy():
                        pygame.mixer.music.load("10.mp3")
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy():
                            time.sleep(0.1)

        # Display (optional)
        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nDetection stopped by user.")
    speak("Detection stopped")

finally:
    cv2.destroyAllWindows()
    picam2.close()
    pygame.mixer.quit()
    pygame.quit()
