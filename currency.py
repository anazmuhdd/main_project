import cv2
import time
from ultralytics import YOLO
from picamera2 import Picamera2
import RPi.GPIO as GPIO
import pygame

# GPIO setup
GPIO.setmode(GPIO.BCM)
PIN_50 = 14
PIN_100 = 23
GPIO.setup(PIN_50, GPIO.OUT)
GPIO.setup(PIN_100, GPIO.OUT)

# Load YOLO model
model_path = './best.pt'
model = YOLO(model_path)
class_names = model.names

# Detection threshold
threshold = 0.5

# Initialize PiCamera2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (1280, 720)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Initialize pygame once
pygame.init()
pygame.mixer.init()

# Allow camera to warm up
time.sleep(0.1)
print("Starting detection... Press Ctrl+C to stop.")

try:
    while True:
        frame = picam2.capture_array()
        results = model(frame)[0]

        for box in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = box
            if score >= threshold:
                label = class_names[int(class_id)]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {score:.2f}", (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if label == '50':
                    GPIO.output(PIN_50, GPIO.HIGH)
                    if not pygame.mixer.music.get_busy():
                        pygame.mixer.music.load("50.mp3")
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy():
                            time.sleep(0.1)
                    GPIO.output(PIN_50, GPIO.LOW)

                elif label == '20':
                   if not pygame.mixer.music.get_busy():
                        pygame.mixer.music.load("20.mp3")
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy():
                            time.sleep(0.1)

        # Show the frame (optional if you're using a GUI)
        cv2.imshow("YOLO Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

except KeyboardInterrupt:
    print("\nDetection stopped by user.")

finally:
    # Cleanup
    cv2.destroyAllWindows()
    picam2.close()
    GPIO.cleanup()
    pygame.mixer.quit()
    pygame.quit()

