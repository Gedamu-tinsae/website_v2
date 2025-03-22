from fastapi import APIRouter, WebSocket
import cv2
import numpy as np
import base64
import json
import tensorflow as tf
import logging
import asyncio
from ...models.tensorflow_model import model
from ...models.plate_correction import extract_text_from_plate

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()

@router.websocket("/ws/realtime-detection")
async def realtime_detection(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            # Receive the base64 encoded frame from frontend
            data = await websocket.receive_text()
            
            try:
                # Decode base64 image
                encoded_data = data.split(',')[1] if ',' in data else data
                nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    continue

                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Prepare input tensor
                input_tensor = tf.convert_to_tensor(rgb_frame)
                input_tensor = input_tensor[tf.newaxis, ...]
                
                # Run detection
                detections = model(input_tensor)
                
                # Process detections
                boxes = detections['detection_boxes'][0].numpy()
                scores = detections['detection_scores'][0].numpy()
                
                height, width, _ = frame.shape
                results = []
                
                # Filter detections with confidence > 0.5
                for i in range(len(scores)):
                    if scores[i] > 0.5:
                        box = boxes[i]
                        # Convert normalized coordinates to pixel coordinates
                        y1, x1, y2, x2 = box
                        x1 = int(x1 * width)
                        x2 = int(x2 * width)
                        y1 = int(y1 * height)
                        y2 = int(y2 * height)
                        
                        # Extract plate region
                        plate_region = frame[y1:y2, x1:x2]
                        if plate_region.size > 0:
                            # Extract text from plate with candidates using our centralized function
                            plate_text, text_candidates = extract_text_from_plate(plate_region, preprocessing_level='minimal')
                            
                            results.append({
                                'bbox': [x1, y1, x2, y2],
                                'confidence': float(scores[i]),
                                'text': plate_text,
                                'text_candidates': text_candidates[:3],  # Include top 3 for realtime (performance)
                                'pattern_match': text_candidates[0].get("pattern_match", False) if text_candidates else False
                            })
                
                # Send results back to client
                await websocket.send_json({
                    'success': True,
                    'detections': results
                })
                
                # Add a small delay to prevent overwhelming the client
                await asyncio.sleep(0.05)
                
            except Exception as e:
                logger.error(f"Error in realtime detection: {e}")
                await websocket.send_json({
                    'success': False,
                    'error': str(e)
                })
                
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        try:
            await websocket.close()
        except:
            pass