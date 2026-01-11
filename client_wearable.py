import cv2
import time
import asyncio
import websockets
import json
import threading
import pygame
import pyttsx3
import numpy as np
from picamera2 import Picamera2
from gpiozero import Button

# -------------------- Configuration --------------------
SERVER_IP = "172.20.10.3"  # REPLACE with your Laptop's IP Address
SERVER_PORT = 8000
WS_URL = f"ws://{SERVER_IP}:{SERVER_PORT}/ws/detect"

# -------------------- Audio Setup --------------------
pygame.init()
pygame.mixer.init()

engine = pyttsx3.init()
engine.setProperty('rate', 130)
speak_lock = threading.Lock()

def speak(text):
    """Speak text safely from a separate thread."""
    def _speak():
        with speak_lock:
            engine.say(text)
            engine.runAndWait()
    threading.Thread(target=_speak, daemon=True).start()

def play_audio_file(filename):
    """Play an MP3 file safely."""
    def _play():
        if not pygame.mixer.music.get_busy():
            try:
                pygame.mixer.music.load(filename)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
            except Exception as e:
                print(f"Error playing audio: {e}")
    threading.Thread(target=_play, daemon=True).start()

# -------------------- Camera Setup (PRESERVED) --------------------
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
# Note: picam2.configure("preview") might need a display, but for streaming we might want "null" or just capture.
# The original code used "preview" and start(). We will keep it but capture arrays.
try:
    picam2.configure("preview")
    picam2.start()
    time.sleep(2) # Warmup
    print("Camera started.")
except Exception as e:
    print(f"Warning: Camera preview config failed (headless?): {e}")
    # Fallback to no preview if needed, or just proceed if it's just a display error
    
# -------------------- Button & State --------------------
BUTTON_PIN = 26
try:
    button = Button(BUTTON_PIN, pull_up=True)
except Exception as e:
    print(f"Button init failed (not on Pi?): {e}")
    button = None

current_mode = "currency"
mode_changed = False

def toggle_mode():
    global current_mode, mode_changed
    current_mode = "object" if current_mode == "currency" else "currency"
    mode_changed = True
    print(f"Button pressed! Switching to {current_mode}...")
    speak(f"Switched to {current_mode} mode")

if button:
    button.when_pressed = toggle_mode

# -------------------- Main Loop --------------------
async def send_frames():
    global mode_changed
    
    print(f"Connecting to {WS_URL}...")
    async with websockets.connect(WS_URL) as websocket:
        print("Connected to Server!")
        speak("Connected to server")
        
        while True:
            # 1. Capture Frame
            frame = picam2.capture_array()
            
            # 2. Encode to JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            bytes_data = buffer.tobytes()
            
            # 3. Check for Mode Change
            if mode_changed:
                await websocket.send(f"MODE:{current_mode}")
                mode_changed = False
            
            # 4. Send Frame
            await websocket.send(bytes_data)
            
            # 5. Receive Results (Non-blocking wait for response? No, we wait for inference)
            # Depending on latency, we might want to send/receive concurrently. 
            # For simplicity, strict step-lock (Send -> Wait Reply) ensures we don't flood.
            try:
                response = await websocket.recv()
                data = json.loads(response)
                
                detections = data.get("detections", [])
                server_mode = data.get("mode", "currency")
                
                # Feedback Logic
                if detections:
                    # Just pick the highest score?
                    detections.sort(key=lambda x: x["score"], reverse=True)
                    top_result = detections[0]
                    label = top_result["label"]
                    score = top_result["score"]
                    
                    print(f"Detected: {label} ({score:.2f})")
                    
                    if server_mode == "currency":
                        # Specific Audio Files for Currency
                        if label in ['10', '20', '50', '100', '500']:
                             play_audio_file(f"{label}.mp3")
                        else:
                             speak(label)
                    else:
                        # General Object
                        speak(label)
                        
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed")
                break
            
            # Optional: Limit FPS if needed
            await asyncio.sleep(0.01)

if __name__ == "__main__":
    try:
        asyncio.run(send_frames())
    except KeyboardInterrupt:
        print("Stopped by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        picam2.stop()
        picam2.close()
