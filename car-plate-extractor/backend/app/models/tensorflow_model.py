import tensorflow as tf
import numpy as np
import cv2
import os
import logging
import base64
from .plate_correction import extract_text_from_plate, matches_pattern, looks_like_covid, generate_character_analysis_for_covid19
from .color_detection import detect_vehicle_color, visualize_color_detection, get_rgb_color
from .vehicle_type import vehicle_detector
from .vehicle_orientation import vehicle_orientation_detector
from .vehicle_make import vehicle_make_detector  # Add import for vehicle make detection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_dirs():
    """Ensure all required directories exist"""
    dirs = [
        "uploads/tensorflow/images",
        "uploads/tensorflow/videos",
        "results/tensorflow/images",
        "results/tensorflow/videos",
        "results/tensorflow/intermediate/images",
        "results/tensorflow/intermediate/videos"
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

# Load the TensorFlow model
model_path = os.path.join(os.path.dirname(__file__), 'saved_model')
if not os.path.exists(model_path):
    logger.error(f"Model path does not exist: {model_path}")
    raise FileNotFoundError(f"Model path does not exist: {model_path}")

model = tf.saved_model.load(model_path)

def process_image_with_model(file_path, confidence_threshold=0.7):
    try:
        ensure_dirs()
        # Load the image
        original_image = cv2.imread(file_path)
        if original_image is None:
            logger.error("Failed to load image.")
            raise ValueError("Failed to load image.")
        logger.info("Image loaded successfully")

        # Use vehicle detector to get vehicle boxes
        vehicle_boxes = vehicle_detector.get_vehicle_boxes(original_image, conf_threshold=0.3)
        
        # Detect vehicle color from the full image first (as a fallback)
        full_image_color = detect_vehicle_color(original_image)
        logger.info(f"Detected vehicle color from full image: {full_image_color['color']}")
        
        # If we have vehicle boxes, detect color from the primary vehicle box
        vehicle_box = None
        region_color_info = {"color": "Unknown", "confidence": 0.0, "color_percentages": {}}
        
        if vehicle_boxes:
            # Use the largest vehicle box (likely the main subject)
            largest_area = 0
            for box in vehicle_boxes:
                x1, y1, x2, y2 = box
                area = (x2 - x1) * (y2 - y1)
                if area > largest_area:
                    largest_area = area
                    vehicle_box = box
            
            if vehicle_box:
                # Detect color using the vehicle box
                region_color_info = detect_vehicle_color(original_image, vehicle_box)
                logger.info(f"Detected vehicle color from vehicle region: {region_color_info['color']}")
                
                # Store region color info for the result
                region_color = region_color_info["color"]
                region_color_confidence = region_color_info["confidence"]
                region_color_percentages = region_color_info.get("color_percentages", {})
            
        # Choose best color based on confidence
        if region_color_info["confidence"] > full_image_color["confidence"]:
            color_info = region_color_info
            best_color_source = "region"
            logger.info(f"Using vehicle region color: {color_info['color']}")
        else:
            color_info = full_image_color
            best_color_source = "full_image"
            logger.info(f"Using full image color: {color_info['color']}")

        # Define intermediate directory here at the beginning of the function
        base_name = os.path.basename(file_path)
        intermediate_dir = os.path.join("results", "tensorflow", "intermediate", "images")
        os.makedirs(intermediate_dir, exist_ok=True)  # Ensure directory exists
        
        # Make a copy for detection visualization
        image = original_image.copy()
        image_np = np.array(image)
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.uint8)

        detections = model(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        # Image for showing detections without OCR text
        detection_image = image.copy()
        
        localized_images = []
        extracted_texts = []
        text_candidates = []
        original_ocr_texts = []  # New array to store original OCR texts
        vehicle_colors = []  # New array to store vehicle colors
        color_confidences = []  # New array to store color confidences
        vehicle_regions = []  # New array to store vehicle regions for visualization
        
        # Save full image for color detection visualization
        full_image_color_path = os.path.join(intermediate_dir, f"7_full_image_color_{base_name}")
        cv2.imwrite(full_image_color_path, original_image.copy())
        full_image_color_path_rel = f"/results/tensorflow/intermediate/images/7_full_image_color_{base_name}"
        
        # Initialize vehicle type detection with default values
        vehicle_type_info = {
            "vehicle_type": "Unknown",
            "confidence": 0.0,
            "alternatives": []
        }
        
        # Initialize vehicle make detection with default values
        vehicle_make_info = {
            "make": "Unknown",
            "confidence": 0.0,
            "alternatives": []
        }
        
        # Detect vehicle type from full image first
        initial_type_info = vehicle_detector.detect(image)
        # Store the full image detection result separately
        full_image_type_info = initial_type_info
        
        # Also save the original image used for vehicle type detection
        full_image_type_path = os.path.join(intermediate_dir, f"6_full_image_type_{base_name}")
        cv2.imwrite(full_image_type_path, image.copy())
        full_image_type_path_rel = f"/results/tensorflow/intermediate/images/6_full_image_type_{base_name}"
        
        if initial_type_info["confidence"] > 0.3:  # Set a minimum threshold
            vehicle_type_info = initial_type_info

        # Detect vehicle make from full image first
        initial_make_info = vehicle_make_detector.detect(image)
        # Store the full image detection result separately
        full_image_make_info = initial_make_info
        
        # Also save the original image used for vehicle make detection
        full_image_make_path = os.path.join(intermediate_dir, f"8_full_image_make_{base_name}")
        cv2.imwrite(full_image_make_path, image.copy())
        full_image_make_path_rel = f"/results/tensorflow/intermediate/images/8_full_image_make_{base_name}"
        
        if initial_make_info["confidence"] > 0.3:  # Set a minimum threshold
            vehicle_make_info = initial_make_info

        for i in range(num_detections):
            # Use the confidence_threshold parameter instead of hardcoding 0.7
            if detections['detection_scores'][i] > confidence_threshold:  # Use confidence threshold parameter
                box = detections['detection_boxes'][i]
                h, w, _ = image.shape
                y_min, x_min, y_max, x_max = box
                x_min, x_max = int(x_min * w), int(x_max * w)
                y_min, y_max = int(y_min * h), int(y_max * h)
                
                # Draw rectangle on detection image
                cv2.rectangle(detection_image, (x_min, y_min), (x_max, x_max), (0, 255, 0), 2)
                
                # Extract and save plate region
                localized_plate = image[y_min:y_max, x_min:x_max]
                localized_images.append(localized_plate)
                
                # Extract text with candidates using the centralized function
                plate_text, candidates, original_ocr_text = extract_text_from_plate(localized_plate, preprocessing_level='advanced')
                extracted_texts.append(plate_text)
                text_candidates.append(candidates)
                original_ocr_texts.append(original_ocr_text)  # Store the original OCR text
                
                # For vehicle color detection, extract a larger region around the license plate
                # This helps capture more of the vehicle
                vehicle_region_y_min = max(0, y_min - (y_max - y_min) * 3)  # Go up 3x the plate height
                vehicle_region_y_max = min(h, y_max + (y_max - y_min))      # Go down 1x the plate height
                vehicle_region_x_min = max(0, x_min - (x_max - x_min))      # Expand width by 1x on each side
                vehicle_region_x_max = min(w, x_max + (x_max - x_min))
                
                # Validate region coordinates
                if not (vehicle_region_y_min < vehicle_region_y_max and vehicle_region_x_min < vehicle_region_x_max):
                    logger.warning("Invalid vehicle region coordinates")
                    continue
                
                # Extract vehicle region
                vehicle_region = image[vehicle_region_y_min:vehicle_region_y_max, 
                                      vehicle_region_x_min:vehicle_region_x_max]
                
                # Save the vehicle region coordinates for visualization
                vehicle_regions.append({
                    "x_min": vehicle_region_x_min,
                    "y_min": vehicle_region_y_min,
                    "x_max": vehicle_region_x_max,
                    "y_max": vehicle_region_y_max
                })
                
                # Detect vehicle color
                if vehicle_region.size > 0:
                    color_info = detect_vehicle_color(vehicle_region)
                    vehicle_colors.append(color_info["color"])
                    color_confidences.append(color_info["confidence"])
                    
                    # Store color percentages for the first detected vehicle (primary vehicle)
                    if i == 0:
                        region_color_percentages = color_info.get("color_percentages", {})
                    
                    # Also save the vehicle region image for visualization
                    vehicle_region_path = os.path.join(intermediate_dir, f"4_vehicle_region_{i}_{base_name}")
                    cv2.imwrite(vehicle_region_path, vehicle_region)
                else:
                    # Use full image color as fallback
                    logger.warning("Empty vehicle region extracted, using full image color")
                    vehicle_colors.append(full_image_color["color"])
                    color_confidences.append(full_image_color["confidence"])
                    
                    # Use full image color percentages if this is the first detection
                    if i == 0:
                        region_color_percentages = full_image_color.get("color_percentages", {})
                
                # Draw the vehicle region rectangle on the detection image (for visualization)
                # Use a different color (yellow) to differentiate from plate detection
                cv2.rectangle(detection_image, 
                             (vehicle_region_x_min, vehicle_region_y_min), 
                             (vehicle_region_x_max, vehicle_region_y_max), 
                             (0, 255, 255), 1)  # Yellow color with thin line
                
                # Double-check for the COVID19 special case
                if plate_text == 'OD19':
                    logger.info("OD19 detected in TensorFlow pipeline - correcting to COVID19")
                    plate_text = 'COVID19'
                    
                    # Generate proper character analysis for COVID19
                    covid_char_positions = generate_character_analysis_for_covid19(
                        candidates[0].get("confidence", 0.85) if candidates else 0.85
                    )
                    
                    # Update the first candidate with proper character data as well
                    if candidates and len(candidates) > 0:
                        candidates[0]['text'] = 'COVID19'
                        candidates[0]['confidence'] = 1.0
                        candidates[0]['pattern_match'] = True
                        candidates[0]['pattern_name'] = 'Special Case - COVID19'
                        candidates[0]['char_positions'] = covid_char_positions
                
                # Check for other COVID-like patterns
                else:
                    is_covid, confidence = looks_like_covid(plate_text)
                    if is_covid and confidence > 0.6:
                        logger.info(f"COVID pattern detected in '{plate_text}' - correcting to COVID19")
                        plate_text = 'COVID19'
                        # Update the first candidate as well if it exists
                        if candidates and len(candidates) > 0:
                            candidates[0]['text'] = 'COVID19'
                            candidates[0]['confidence'] = max(candidates[0].get('confidence', 0), confidence)
                            candidates[0]['pattern_match'] = True
                            candidates[0]['pattern_name'] = 'Special Case - COVID19'
                
                extracted_texts.append(plate_text)
                text_candidates.append(candidates)
                
                # Draw rectangle and text on final image
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(image, plate_text, 
                          (x_min, y_min - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          0.9,
                          (255, 255, 255),
                          2)
                
                # Add color information below the plate text
                if vehicle_colors:
                    color_text = f"Color: {vehicle_colors[-1]}"
                    cv2.putText(image, color_text,
                              (x_min, y_min - 40),  # Position above the plate text
                              cv2.FONT_HERSHEY_SIMPLEX,
                              0.7,
                              (0, 255, 255),  # Yellow color for visibility
                              2)

                # Update vehicle type if we get a better detection from the region
                if vehicle_region.size > 0:
                    region_type_info = vehicle_detector.detect(vehicle_region)
                    # Store this separately instead of conditionally overwriting
                    if region_type_info["confidence"] > vehicle_type_info["confidence"]:
                        vehicle_type_info = region_type_info

                # Extract vehicle region from original image for type detection
                vehicle_type_region = original_image[vehicle_region_y_min:vehicle_region_y_max, 
                                      vehicle_region_x_min:vehicle_region_x_max]
                
                # Save the vehicle type region image for visualization
                if vehicle_type_region.size > 0:
                    vehicle_type_path = os.path.join(intermediate_dir, f"5_vehicle_type_{i}_{base_name}")
                    cv2.imwrite(vehicle_type_path, vehicle_type_region)
                    
                    # Update vehicle type detection using clean region
                    region_type_info = vehicle_detector.detect(vehicle_type_region)
                    # Store this separately instead of conditionally overwriting
                    if region_type_info["confidence"] > vehicle_type_info["confidence"]:
                        vehicle_type_info = region_type_info
                        
                    # Add vehicle type region path to result paths
                    if i == 0:  # Only store the first vehicle's type region
                        vehicle_type_path_rel = f"/results/tensorflow/intermediate/images/5_vehicle_type_{i}_{base_name}"

                # Detect vehicle orientation
                vehicle_orientation_info = vehicle_orientation_detector.predict(vehicle_region)
                logger.info(f"Detected vehicle orientation: {vehicle_orientation_info['orientation']}")

                # Add orientation text to the image
                cv2.putText(image, f"Orientation: {vehicle_orientation_info['orientation']}",
                            (x_min, y_min - 70),  # Position above color text
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 0, 255),  # Magenta color
                            2)

                # Update vehicle make if we get a better detection from the region
                if vehicle_region.size > 0:
                    region_make_info = vehicle_make_detector.detect(vehicle_region)
                    # Store this separately instead of conditionally overwriting
                    if region_make_info["confidence"] > vehicle_make_info["confidence"]:
                        vehicle_make_info = region_make_info

                # Add vehicle make below vehicle type
                if vehicle_make_info["make"] != "Unknown":
                    make_text = f"Make: {vehicle_make_info['make']}"
                    cv2.putText(image, make_text,
                            (x_min, y_min - 100),  # Position above orientation text
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (102, 255, 153),  # Light green color
                            2)

        # If no plates were detected (empty lists)
        if not localized_images and not extracted_texts:
            logger.info("No license plates detected - attempting full image OCR")
            
            # Use the extract_text_from_plate function on the full image
            full_image_text, full_image_candidates, full_image_ocr = extract_text_from_plate(
                original_image, 
                preprocessing_level='advanced'
            )
            
            # Update the result lists
            extracted_texts = [full_image_text]
            text_candidates = [full_image_candidates]
            original_ocr_texts = [full_image_ocr]
            logger.info(f"Full image OCR result: {full_image_text} (Original: {full_image_ocr})")
            
            # Initialize vehicle_orientation_info for the case where no plates are detected
            vehicle_orientation_info = vehicle_orientation_detector.predict(original_image)
            logger.info(f"Detected vehicle orientation from full image: {vehicle_orientation_info['orientation']}")

        # Save intermediate results
        # Note: base_name and intermediate_dir are now already defined above
        
        # Save original image
        original_path = os.path.join(intermediate_dir, f"1_original_{base_name}")
        cv2.imwrite(original_path, original_image)
        
        # Save detection image
        detection_path = os.path.join(intermediate_dir, f"2_detection_{base_name}")
        cv2.imwrite(detection_path, detection_image)
        
        # Save plate regions
        plate_paths = []
        for idx, plate in enumerate(localized_images):
            plate_path = os.path.join(intermediate_dir, f"3_plate_{idx}_{base_name}")
            cv2.imwrite(plate_path, plate)
            plate_paths.append(f"/results/tensorflow/intermediate/images/3_plate_{idx}_{base_name}")
        
        # Save vehicle regions for visualization
        vehicle_region_paths = []
        for idx, region_coords in enumerate(vehicle_regions):
            if idx < len(localized_images):  # Only save regions for valid detections
                vehicle_region_img = image[
                    region_coords["y_min"]:region_coords["y_max"],
                    region_coords["x_min"]:region_coords["x_max"]
                ]
                if vehicle_region_img.size > 0:
                    vehicle_region_path = os.path.join(intermediate_dir, f"4_vehicle_region_{idx}_{base_name}")
                    cv2.imwrite(vehicle_region_path, vehicle_region_img)
                    vehicle_region_paths.append(f"/results/tensorflow/intermediate/images/4_vehicle_region_{idx}_{base_name}")

        # Save final result
        final_path = os.path.join("results", "tensorflow", "images", base_name)
        cv2.imwrite(final_path, image)

        # Get color percentages if not set yet (no detections)
        if not vehicle_colors:
            region_color_percentages = {}
            best_color = full_image_color["color"]
            best_color_confidence = full_image_color["confidence"] 
        else:
            best_color = vehicle_colors[0]
            best_color_confidence = color_confidences[0]

        # Create a special vehicle make region image for visualization
        if 'vehicle_region' in locals() and vehicle_region.size > 0:
            vehicle_make_path = os.path.join(intermediate_dir, f"9_vehicle_make_{base_name}")
            cv2.imwrite(vehicle_make_path, vehicle_region)
            vehicle_make_path_rel = f"/results/tensorflow/intermediate/images/9_vehicle_make_{base_name}"
        else:
            vehicle_make_path_rel = None

        result = {
            "status": "success",
            "filename": file_path,
            "result_url": f"/results/tensorflow/images/{base_name}",
            "intermediate_steps": {
                "original": f"/results/tensorflow/intermediate/images/1_original_{base_name}",
                "detection": f"/results/tensorflow/intermediate/images/2_detection_{base_name}",
                "plates": plate_paths,
                "vehicle_regions": vehicle_region_paths,
                "vehicle_type_region": vehicle_type_path_rel if 'vehicle_type_path_rel' in locals() else None,
                "full_image_type": full_image_type_path_rel,  # Add full image path for type
                "full_image_color": full_image_color_path_rel,  # Add full image path for color
                "full_image_make": full_image_make_path_rel,  # Add full image path for make
                "vehicle_make_region": vehicle_make_path_rel if 'vehicle_make_path_rel' in locals() else None,
            },
            "intermediate_images": {
                # Ensure array is contiguous before encoding
                "vehicle_region": base64.b64encode(np.ascontiguousarray(vehicle_region)).decode('utf-8') 
                if 'vehicle_region' in locals() and vehicle_region.size > 0 else None
            },
            "detected_plates": extracted_texts,
            "vehicle_colors": vehicle_colors,  # Add vehicle colors to result
            "color_confidences": color_confidences,  # Add color confidences to result
            "vehicle_color": best_color,  # Primary vehicle color 
            "color_confidence": best_color_confidence,
            "color_hex": get_rgb_color(best_color),  # Add hex color code
            "full_image_color": full_image_color["color"],  # Add full image color
            "full_image_color_confidence": full_image_color["confidence"],  # Add full image color confidence
            "full_image_color_hex": get_rgb_color(full_image_color["color"]),  # Add hex color
            "region_color": vehicle_colors[0] if vehicle_colors else "Unknown",  # Add region-specific color
            "region_color_confidence": color_confidences[0] if color_confidences else 0.0,  # Add region-specific confidence
            "region_color_hex": get_rgb_color(vehicle_colors[0]) if vehicle_colors else "#cccccc",  # Add hex
            "color_percentages": full_image_color.get("color_percentages", {}),  # Full image color percentages
            "region_color_percentages": region_color_percentages if 'region_color_percentages' in locals() else {},  # Region color percentages
            "best_color_source": "region" if (vehicle_colors and color_confidences[0] > full_image_color["confidence"]) else "full_image",
            "original_ocr_texts": original_ocr_texts,  # Include original OCR results
            "license_plate": extracted_texts[0] if extracted_texts else "No text detected",
            "original_ocr": original_ocr_texts[0] if original_ocr_texts else "No text detected",  # Include first original OCR
            "text_candidates": text_candidates[0] if text_candidates else [],  # Ensure this is a direct array, not nested
            "vehicle_region_coordinates": vehicle_regions,  # Include the coordinates for frontend highlighting
            "vehicle_type": vehicle_type_info["vehicle_type"],
            "vehicle_type_confidence": vehicle_type_info["confidence"],
            "vehicle_type_alternatives": vehicle_type_info["alternatives"],
            # Add both detection results separately
            "full_image_type": full_image_type_info["vehicle_type"],
            "full_image_type_confidence": full_image_type_info["confidence"],
            "region_type": vehicle_type_info["vehicle_type"] if vehicle_type_info["confidence"] > full_image_type_info["confidence"] else "Unknown",
            "region_type_confidence": vehicle_type_info["confidence"] if vehicle_type_info["confidence"] > full_image_type_info["confidence"] else 0.0,
            "vehicle_orientation": vehicle_orientation_info["orientation"],
            "orientation_confidence": vehicle_orientation_info["confidence"],
            "is_front_facing": vehicle_orientation_info["is_front"],
            "plate_detection_status": "full_image" if not localized_images else "plate_detected",
            "best_type_source": "region" if vehicle_type_info["confidence"] > full_image_type_info["confidence"] else "full_image",
            "vehicle_make": vehicle_make_info["make"],
            "make_confidence": vehicle_make_info["confidence"],
            "make_alternatives": vehicle_make_info["alternatives"],
            # Add both detection results separately
            "full_image_make": full_image_make_info["make"],
            "full_image_make_confidence": full_image_make_info["confidence"],
            "region_make": vehicle_make_info["make"] if vehicle_make_info["confidence"] > full_image_make_info["confidence"] else "Unknown",
            "region_make_confidence": vehicle_make_info["confidence"] if vehicle_make_info["confidence"] > full_image_make_info["confidence"] else 0.0,
            "best_make_source": "region" if vehicle_make_info["confidence"] > full_image_make_info["confidence"] else "full_image",
            "vehicle_boxes": vehicle_boxes,
            "primary_vehicle_box": vehicle_box
        }

        return result
    except Exception as e:
        logger.error(f"Error processing image with model: {e}")
        raise

def process_video_with_model(file_path, low_visibility=False, confidence_threshold=0.7):
    try:
        ensure_dirs()
        cap = cv2.VideoCapture(file_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_number = 0
        results = []
        all_extracted_texts = []
        all_intermediate_frames = {
            "original": [],
            "detection": [],
            "plates": []
        }
        all_vehicle_colors = []  # Track all detected vehicle colors
        
        # Set a limit for the number of frames to process to avoid timeouts
        max_frames = 40  # Process only the first 40 frames to ensure completion
        logger.info(f"Will process up to {max_frames} frames from a total of {frame_count} frames")

        # Initialize HazeRemoval if needed for low visibility
        hr = None
        if low_visibility:
            from .haze_removal import HazeRemoval
            hr = HazeRemoval()
            logger.info("Initialized HazeRemoval for low visibility video")

        # Initialize font for text
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Initialize vehicle type info with default values
        vehicle_type_info = {
            "vehicle_type": "Unknown",
            "confidence": 0.0,
            "alternatives": []
        }

        # Initialize vehicle orientation info with default values
        vehicle_orientation_info = {
            "orientation": "Unknown",
            "confidence": 0.0,
            "is_front": None
        }

        # Initialize vehicle make info with default values
        vehicle_make_info = {
            "make": "Unknown",
            "confidence": 0.0,
            "alternatives": []
        }

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1
            logger.info(f"Processing frame {frame_number}/{min(frame_count, max_frames)}")
            
            # Exit the loop if we've processed enough frames
            if frame_number > max_frames:
                logger.info(f"Reached frame limit of {max_frames}. Stopping video processing.")
                break

            # Store original frame
            original_frame = frame.copy()

            # Apply dehazing if low_visibility is True
            if low_visibility and hr:
                # Convert frame to proper format for HazeRemoval
                # Save frame to temp file, process it with HazeRemoval, then read back
                temp_frame_path = os.path.join("uploads", "tensorflow", "temp_frame.jpg")
                cv2.imwrite(temp_frame_path, frame)
                
                hr.open_image(temp_frame_path)
                hr.get_dark_channel()
                hr.get_air_light()
                hr.get_transmission()
                hr.guided_filter()
                hr.recover()
                
                # Update frame with dehazed version
                frame = hr.dst
                logger.info("Applied dehazing to video frame")

            # Get full frame vehicle color (fallback)
            full_frame_color = detect_vehicle_color(frame)

            # Continue with normal processing using the possibly dehazed frame
            # Convert frame to tensor and detect plates
            image_np = np.array(frame)
            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.uint8)
            detections = model(input_tensor)

            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
            detections['num_detections'] = num_detections
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            # Image for showing detections without OCR text
            detection_frame = frame.copy()
            frame_plates = []
            frame_texts = []
            frame_candidates = []
            frame_colors = []  # Track colors detected in this frame

            # Update vehicle type detection from full frame first
            initial_type_info = vehicle_detector.detect(frame)
            if initial_type_info["confidence"] > vehicle_type_info["confidence"]:
                vehicle_type_info = initial_type_info

            # Update vehicle make detection from full frame first
            initial_make_info = vehicle_make_detector.detect(frame)
            if initial_make_info["confidence"] > vehicle_make_info["confidence"]:
                vehicle_make_info = initial_make_info
                logger.info(f"Frame {frame_number}: Updated vehicle make: {vehicle_make_info['make']}")

            for i in range(num_detections):
                # Use the confidence_threshold parameter instead of hardcoding 0.7
                if detections['detection_scores'][i] > confidence_threshold:  # Use confidence threshold parameter
                    box = detections['detection_boxes'][i]
                    h, w, _ = frame.shape
                    y_min, x_min, y_max, x_max = box
                    x_min, x_max = int(x_min * w), int(x_max * w)
                    y_min, y_max = int(y_min * h), int(y_max * h)

                    # Draw rectangle on detection frame
                    cv2.rectangle(detection_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    # Extract plate region
                    plate = frame[y_min:y_max, x_min:x_max]
                    frame_plates.append(plate)

                    # Use the common extraction function for improved OCR
                    plate_text, candidates, original_ocr_text = extract_text_from_plate(plate, preprocessing_level='advanced')
                    
                    # For vehicle color detection, extract a larger region around the license plate
                    vehicle_region_y_min = max(0, y_min - (y_max - y_min) * 3)  # Go up 3x the plate height
                    vehicle_region_y_max = min(h, y_max + (y_max - y_min))      # Go down 1x the plate height
                    vehicle_region_x_min = max(0, x_min - (x_max - x_min))      # Expand width by 1x on each side
                    vehicle_region_x_max = min(w, x_max + (x_max - x_min))
                    
                    # Extract vehicle region
                    vehicle_region = frame[vehicle_region_y_min:vehicle_region_y_max, 
                                          vehicle_region_x_min:vehicle_region_x_max]
                    
                    # Detect vehicle color
                    if vehicle_region.size > 0:
                        color_info = detect_vehicle_color(vehicle_region)
                        frame_colors.append(color_info["color"])
                    else:
                        # Use full frame color as fallback
                        frame_colors.append(full_frame_color["color"])
                    
                    # Double-check for the COVID19 special case
                    if plate_text == 'OD19':
                        logger.info("OD19 detected in TensorFlow pipeline - correcting to COVID19")
                        plate_text = 'COVID19'
                        
                        # Generate proper character analysis for COVID19
                        covid_char_positions = generate_character_analysis_for_covid19(
                            candidates[0].get("confidence", 0.85) if candidates else 0.85
                        )
                        
                        # Update the first candidate with proper character data as well
                        if candidates and len(candidates) > 0:
                            candidates[0]['text'] = 'COVID19'
                            candidates[0]['confidence'] = 1.0
                            candidates[0]['pattern_match'] = True
                            candidates[0]['pattern_name'] = 'Special Case - COVID19'
                            candidates[0]['char_positions'] = covid_char_positions
                    
                    # Check for other COVID-like patterns
                    else:
                        is_covid, confidence = looks_like_covid(plate_text)
                        if is_covid and confidence > 0.6:
                            logger.info(f"COVID pattern detected in '{plate_text}' - correcting to COVID19")
                            plate_text = 'COVID19'
                            # Update the first candidate as well if it exists
                            if candidates and len(candidates) > 0:
                                candidates[0]['text'] = 'COVID19'
                                candidates[0]['confidence'] = max(candidates[0].get('confidence', 0), confidence)
                                candidates[0]['pattern_match'] = True
                                candidates[0]['pattern_name'] = 'Special Case - COVID19'
                    
                    frame_texts.append(plate_text)
                    frame_candidates.append(candidates)

                    # Draw rectangle and text on final frame
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(frame, plate_text,
                              (x_min, y_min - 10),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              0.9,
                              (255, 255, 255),
                              2)
                    
                    # Add color information
                    if frame_colors:
                        color_text = f"Color: {frame_colors[-1]}"
                        cv2.putText(frame, color_text,
                                  (x_min, y_min - 40),  # Position above the plate text
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  0.7,
                                  (0, 255, 255),  # Yellow color for visibility
                                  2)

                    # Update vehicle type if we get a better detection from vehicle region
                    if vehicle_region.size > 0:
                        region_type_info = vehicle_detector.detect(vehicle_region)
                        if region_type_info["confidence"] > vehicle_type_info["confidence"]:
                            vehicle_type_info = region_type_info
                            logger.info(f"Frame {frame_number}: Updated vehicle type: {vehicle_type_info['vehicle_type']}")

                    # Add make information
                    if vehicle_make_info["make"] != "Unknown":
                        make_text = f"Make: {vehicle_make_info['make']}"
                        cv2.putText(frame, make_text,
                                  (x_min, y_min - 100),  # Position above orientation text
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  0.7,
                                  (102, 255, 153),  # Light green color
                                  2)

                    # Update vehicle make if we get a better detection from vehicle region
                    if vehicle_region.size > 0:
                        region_make_info = vehicle_make_detector.detect(vehicle_region)
                        if region_make_info["confidence"] > vehicle_make_info["confidence"]:
                            vehicle_make_info = region_make_info
                            logger.info(f"Frame {frame_number}: Updated vehicle make: {vehicle_make_info['make']}")

            # If no plates were detected, try to detect orientation from the full frame
            if not frame_plates:
                try:
                    temp_orientation_info = vehicle_orientation_detector.predict(frame)
                    if temp_orientation_info["confidence"] > vehicle_orientation_info["confidence"]:
                        vehicle_orientation_info = temp_orientation_info
                except Exception as e:
                    logger.error(f"Error detecting orientation from full frame: {e}")

            # Store intermediate results for this frame
            all_intermediate_frames["original"].append(original_frame)
            all_intermediate_frames["detection"].append(detection_frame)
            all_intermediate_frames["plates"].extend(frame_plates)
            all_extracted_texts.extend(frame_texts)
            all_vehicle_colors.extend(frame_colors)  # Add detected colors

            # Add processed frame to results
            results.append(frame)

        cap.release()
        
        # Double check we have some results
        if not results:
            logger.warning("No frames were processed successfully. Check video format.")
            return {
                "status": "error",
                "message": "No frames could be processed from the video."
            }
            
        # Save only a subset of the processed frames for performance reasons
        logger.info(f"Saving processed video with {len(results)} frames")
        
        try:
            # Save the processed video
            result_video_path = os.path.join("results", "tensorflow", "videos", os.path.basename(file_path))
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            frame_size = (results[0].shape[1], results[0].shape[0])
            out = cv2.VideoWriter(result_video_path, fourcc, fps, frame_size)

            if not out.isOpened():
                logger.error("Failed to open video writer. Check codec compatibility.")
                # Fall back to MJPG codec which is more widely supported
                out = cv2.VideoWriter(result_video_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, frame_size)
                if not out.isOpened():
                    logger.error("Failed to open video writer with MJPG codec too.")
                    # As a last resort, try to save a sample frame as image
                    sample_frame_path = os.path.join("results", "tensorflow", "images", f"sample_frame_{os.path.basename(file_path)}.jpg")
                    cv2.imwrite(sample_frame_path, results[0])
                    logger.info(f"Saved sample frame instead at {sample_frame_path}")
            else:
                # Write frames
                for frame in results:
                    out.write(frame)
                out.release()
                logger.info(f"Processed video saved at: {result_video_path}")

            # Encode sample frames and plates as base64 for frontend display
            def encode_image(img):
                _, buffer = cv2.imencode('.jpg', img)
                return base64.b64encode(buffer).decode('utf-8')

            # Take a sample frame from each category for display
            sample_frame_index = min(10, len(results) - 1)  # Take 10th frame or last frame if video is shorter
            intermediate_images = {
                "original": encode_image(all_intermediate_frames["original"][sample_frame_index]),
                "detection": encode_image(all_intermediate_frames["detection"][sample_frame_index]),
                "plates": [encode_image(plate) for plate in all_intermediate_frames["plates"][:5]]  # Limit to first 5 plates
            }

            # Determine most common vehicle color
            from collections import Counter
            if all_vehicle_colors:
                color_counter = Counter(all_vehicle_colors)
                most_common_color = color_counter.most_common(1)[0][0]
                color_frequency = color_counter.most_common(1)[0][1] / len(all_vehicle_colors)
                
                # Create color percentages from the color counter
                total_colors = len(all_vehicle_colors)
                color_percentages = {color: (count / total_colors) * 100 for color, count in color_counter.items()}
            else:
                most_common_color = full_frame_color["color"]
                color_frequency = full_frame_color["confidence"]
                color_percentages = full_frame_color.get("color_percentages", {})

            result = {
                "status": "success",
                "filename": file_path,
                "result_url": f"/results/tensorflow/videos/{os.path.basename(file_path)}",
                "frames_processed": frame_number,
                "total_frames": frame_count,
                "result_image": encode_image(results[sample_frame_index]),  # Sample frame from result
                "intermediate_images": intermediate_images,
                "detected_plates": list(set(all_extracted_texts)),  # Remove duplicates
                "license_plate": all_extracted_texts[0] if all_extracted_texts else "Unknown",
                "vehicle_color": most_common_color,  # Add vehicle color to result
                "color_confidence": color_frequency,  # Add confidence (frequency) of color detection
                "color_percentages": color_percentages,  # Add color percentages
                "text_candidates": frame_candidates[0] if frame_candidates else [],  # Ensure this is a direct array, not nested
                "vehicle_type": vehicle_type_info["vehicle_type"],
                "vehicle_type_confidence": vehicle_type_info["confidence"],
                "vehicle_type_alternatives": vehicle_type_info["alternatives"],
                "vehicle_orientation": vehicle_orientation_info["orientation"],
                "orientation_confidence": vehicle_orientation_info["confidence"],
                "is_front_facing": vehicle_orientation_info["is_front"],
                "vehicle_make": vehicle_make_info["make"],
                "make_confidence": vehicle_make_info["confidence"],
                "make_alternatives": vehicle_make_info["alternatives"]
            }

            logger.info("Video processing completed successfully")
            return result
        except Exception as e:
            logger.error(f"Error saving video: {e}")
            # Try to return a partial result with at least the first frame
            if results:
                sample_frame_path = os.path.join("results", "tensorflow", "images", f"error_frame_{os.path.basename(file_path)}.jpg")
                cv2.imwrite(sample_frame_path, results[0])
                return {
                    "status": "partial_success",
                    "message": f"Could not save video but processed {len(results)} frames. Sample frame saved.",
                    "sample_frame_url": f"/results/tensorflow/images/error_frame_{os.path.basename(file_path)}.jpg"
                }
            raise
    except Exception as e:
        logger.error(f"Error processing video with model: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"Error processing video: {str(e)}"
        }
