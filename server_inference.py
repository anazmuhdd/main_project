import cv2
import numpy as np
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from ultralytics import YOLO
import json

app = FastAPI()

# -------------------- Model Setup --------------------
# Load models once at startup
print("Loading YOLO models...")
try:
    currency_model = YOLO("./best.pt")
    object_model = YOLO("./obj1.pt") # Or 'yolo11n.pt' or check path
    # Pre-warm models if needed
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    # Fallback to standard model if custom ones aren't found for testing purposes?
    # For now, we assume files exist as seen in list_dir.

currency_classes = currency_model.names
object_classes = object_model.names

@app.websocket("/ws/detect")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")
    
    # Default mode
    mode = "currency"
    
    try:
        while True:
            # Protocol: Client sends a raw byte array for image
            # OR a text message for control commands (like "MODE:object")
            
            # We need to handle both. easiest is to assume binary is image, text is command.
            message = await websocket.receive()
            
            if "bytes" in message:
                data = message["bytes"]
                
                # Convert bytes to numpy array
                nparr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    continue

                # Run Inference
                results = None
                class_dict = {}
                
                if mode == "currency":
                    results = currency_model(frame, verbose=False)[0]
                    class_dict = currency_classes
                else:
                    results = object_model(frame, verbose=False)[0]
                    class_dict = object_classes
                
                detections = []
                threshold = 0.5
                
                # Process results
                if results:
                    for box in results.boxes.data.tolist():
                        x1, y1, x2, y2, score, class_id = box
                        if score >= threshold:
                            label = class_dict[int(class_id)]
                            detections.append({
                                "label": label,
                                "score": float(score),
                                "box": [float(x1), float(y1), float(x2), float(y2)]
                            })
                
                # Send back results as JSON
                await websocket.send_json({
                    "detections": detections,
                    "mode": mode
                })
                
            elif "text" in message:
                text = message["text"]
                if text.startswith("MODE:"):
                    new_mode = text.split(":")[1]
                    if new_mode in ["currency", "object"]:
                        mode = new_mode
                        print(f"Mode switched to: {mode}")
                        # Acknowledge mode change
                        await websocket.send_json({"event": "mode_changed", "mode": mode})

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import uvicorn
    # Listen on all interfaces so the Client can connect
    uvicorn.run(app, host="0.0.0.0", port=8000)
