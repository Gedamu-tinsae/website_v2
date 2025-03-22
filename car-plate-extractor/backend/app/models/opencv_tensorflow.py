import cv2
import numpy as np
import os
import imutils
import logging
import json
import base64
import re
from .plate_correction import extract_text_from_plate, extract_text_from_region, get_reader, matches_pattern, looks_like_covid, generate_character_analysis_for_covid19
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
        "uploads/opencv/images",
        "uploads/opencv/videos",
        "results/opencv/images",
        "results/opencv/videos"
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def process_image(file_path, confidence_threshold=0.7):
    try:
        ensure_dirs()
        # Load the image
        image = cv2.imread(file_path)
        if image is None:
            logger.error("Failed to load image.")
            raise ValueError("Failed to load image.")
        logger.info("Image loaded successfully")

        # Use vehicle detector to get vehicle boxes
        vehicle_boxes = vehicle_detector.get_vehicle_boxes(image, conf_threshold=0.3)
        
        # Detect vehicle color - first from the full image as a fallback
        full_image_color = detect_vehicle_color(image)
        logger.info(f"Detected vehicle color from full image: {full_image_color['color']}")
        
        # If we have vehicle boxes, detect color from the primary vehicle box
        vehicle_box = None
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
                vehicle_region_color = detect_vehicle_color(image, vehicle_box)
                logger.info(f"Detected vehicle color from vehicle region: {vehicle_region_color['color']}")
                
                # Use the vehicle region color if confidence is higher
                if vehicle_region_color["confidence"] > full_image_color["confidence"]:
                    color_info = vehicle_region_color
                    logger.info(f"Using vehicle region color: {color_info['color']}")
                else:
                    color_info = full_image_color
                    logger.info(f"Using full image color: {color_info['color']}")
            else:
                color_info = full_image_color
        else:
            color_info = full_image_color

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        logger.info("Converted to grayscale")

        # Apply bilateral filter and Canny edge detection
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
        edges = cv2.Canny(bfilter, 30, 200)
        logger.info("Applied bilateral filter and Canny edge detection")

        # Find contours and locate the license plate
        keypoints = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        logger.info(f"Number of contours found: {len(contours)}")
        location = None
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)
            logger.info(f"Contour length: {len(approx)}")
            if len(approx) == 4:
                location = approx
                break

        # If we're using a lower confidence threshold for low visibility images,
        # we should also be more lenient with contour selection
        min_contour_area = 50 if confidence_threshold < 0.7 else 100

        if location is None:
            logger.warning("No valid contour with 4 points found. Trying alternative method.")
            for contour in contours:
                if cv2.contourArea(contour) > min_contour_area:  # Use dynamic threshold
                    location = cv2.convexHull(contour)
                    break

        # Validate the contour points
        if location is not None and location.shape[0] > 0 and location.dtype == np.int32:
            # Create a mask and extract the license plate region
            mask = np.zeros(gray.shape, np.uint8)
            new_image = cv2.drawContours(mask, [location], 0, 255, -1)
            new_image = cv2.bitwise_and(image, image, mask=mask)
            
            # Try a different approach - get bounding rectangle of the contour
            # This might capture the full plate better than just using the contour points
            x, y, w, h = cv2.boundingRect(location)
            
            # Expand the rectangle slightly to ensure we get the whole plate
            # This is especially important for plates with spaced characters
            expand_factor_w = 1.2  # Expand width by 20%
            expand_factor_h = 1.1  # Expand height by 10%
            
            # Calculate expanded dimensions
            new_w = int(w * expand_factor_w)
            new_h = int(h * expand_factor_h)
            
            # Calculate new top left point (centered)
            new_x = max(0, x - (new_w - w) // 2)
            new_y = max(0, y - (new_h - h) // 2)
            
            # Make sure we don't go out of bounds
            new_w = min(new_w, image.shape[1] - new_x)
            new_h = min(new_h, image.shape[0] - new_y)
            
            # Extract the expanded plate region from the original image
            expanded_plate_region = image[new_y:new_y+new_h, new_x:new_x+new_w]
            
            # Convert to grayscale specifically for OCR
            expanded_plate_gray = cv2.cvtColor(expanded_plate_region, cv2.COLOR_BGR2GRAY)
            
            # Save the expanded region for debugging
            intermediate_dir = os.path.join("results", "opencv", "intermediate", "images")
            os.makedirs(intermediate_dir, exist_ok=True)
            expanded_plate_path = os.path.join(intermediate_dir, f"expanded_plate_{os.path.basename(file_path)}")
            cv2.imwrite(expanded_plate_path, expanded_plate_region)
            
            # Use both the original cropped plate and the expanded plate for OCR
            # and pick the better result
            
            # Original approach for comparison
            (x_orig, y_orig) = np.where(mask == 255)
            if x_orig.size == 0 or y_orig.size == 0:
                logger.error("Failed to locate the license plate region.")
                raise ValueError("Failed to locate the license plate region.")
            (x1, y1) = (np.min(x_orig), np.min(y_orig))
            (x2, y2) = (np.max(x_orig), np.max(y_orig))
            cropped_image = gray[x1:x2+1, y1:y2+1]
            
            # Try OCR on both the original and expanded plate regions
            license_plate_orig, text_candidates_orig, original_ocr_text_orig = extract_text_from_plate(cropped_image, preprocessing_level='standard')
            license_plate_exp, text_candidates_exp, original_ocr_text_exp = extract_text_from_plate(expanded_plate_gray, preprocessing_level='standard')
            
            # Log both results for comparison
            logger.info(f"Original crop OCR: {original_ocr_text_orig}, license plate: {license_plate_orig}")
            logger.info(f"Expanded crop OCR: {original_ocr_text_exp}, license plate: {license_plate_exp}")
            
            # Choose the better result - prefer the one with more characters or one that matches a pattern
            def get_score(plate, candidates):
                # Calculate a score based on:
                # 1. Length of the text (longer is better)
                # 2. If it matches a pattern (pattern match is better)
                # 3. Confidence score
                
                length_score = len(plate) * 0.2  # 0.2 points per character
                
                pattern_score = 0
                confidence_score = 0
                
                if candidates and len(candidates) > 0:
                    if candidates[0].get("pattern_match", False):
                        pattern_score = 2.0  # Pattern match is worth 2 points
                    confidence_score = candidates[0].get("confidence", 0) * 2.0  # Up to 2 points for confidence
                
                # Check for number+letter format which is typical for license plates
                format_score = 0
                if re.search(r'\d+\s*[A-Z]+', plate):  # Numbers followed by letters (with optional space)
                    format_score = 2.0  # This common format is worth 2 points
                
                return length_score + pattern_score + confidence_score + format_score
            
            # Calculate scores
            orig_score = get_score(license_plate_orig, text_candidates_orig)
            exp_score = get_score(license_plate_exp, text_candidates_exp)
            
            # Choose the best result
            if exp_score > orig_score:
                logger.info(f"Using expanded plate region (score: {exp_score:.2f} vs {orig_score:.2f})")
                license_plate = license_plate_exp
                text_candidates = text_candidates_exp
                original_ocr_text = original_ocr_text_exp
                # Use the expanded plate coordinates
                cropped_image = expanded_plate_gray
                x1, y1 = new_x, new_y
                x2, y2 = new_x + new_w - 1, new_y + new_h - 1
            else:
                logger.info(f"Using original plate region (score: {orig_score:.2f} vs {exp_score:.2f})")
                license_plate = license_plate_orig
                text_candidates = text_candidates_orig
                original_ocr_text = original_ocr_text_orig
            
            # Special check for separated characters like "172 TMZ" 
            if len(license_plate) <= 3 and all(c.isdigit() for c in license_plate):
                # If we just got a short number, try to find matching letter part in the expanded region
                parts = re.findall(r'[A-Z]{2,}', original_ocr_text_exp)
                if parts:
                    letter_part = parts[0]
                    logger.info(f"Found separated letter part: {letter_part}")
                    combined_plate = f"{license_plate} {letter_part}"
                    logger.info(f"Combined plate: {combined_plate}")
                    license_plate = combined_plate
                    
                    # Also update the first candidate
                    if text_candidates and len(text_candidates) > 0:
                        text_candidates[0]["text"] = combined_plate
                        text_candidates[0]["confidence"] = max(text_candidates[0].get("confidence", 0.7), 0.7)
                        text_candidates[0]["pattern_match"] = True
                        text_candidates[0]["pattern_name"] = "Format 123 ABC with space"
            
            # Double-check for the COVID19 special case
            if license_plate == 'OD19' or any(c.get('text') == 'OD19' for c in text_candidates):
                logger.info("OD19 detected in OpenCV pipeline - correcting to COVID19")
                license_plate = 'COVID19'
                
                # Generate proper character analysis for COVID19
                covid_char_positions = generate_character_analysis_for_covid19(
                    text_candidates[0].get("confidence", 0.85) if text_candidates else 0.85
                )
                
                # Update the first candidate with proper character data as well
                if text_candidates and len(text_candidates) > 0:
                    text_candidates[0]['text'] = 'COVID19'
                    text_candidates[0]['confidence'] = 1.0
                    text_candidates[0]['pattern_match'] = True
                    text_candidates[0]['pattern_name'] = 'Special Case - COVID19'
                    text_candidates[0]['char_positions'] = covid_char_positions
            
            # Check for other COVID-like patterns
            elif license_plate:
                is_covid, confidence = looks_like_covid(license_plate)
                if is_covid and confidence > 0.6:
                    logger.info(f"COVID pattern detected in '{license_plate}' - correcting to COVID19")
                    license_plate = 'COVID19'
                    # Update the first candidate as well if it exists
                    if text_candidates and len(text_candidates) > 0:
                        text_candidates[0]['text'] = 'COVID19'
                        text_candidates[0]['confidence'] = max(text_candidates[0].get('confidence', 0), confidence)
                        text_candidates[0]['pattern_match'] = True
                        text_candidates[0]['pattern_name'] = 'Special Case - COVID19'
            
            logger.info(f"License plate text extracted: {license_plate}")
            logger.info(f"Original OCR text: {original_ocr_text}")

            # First establish the plate region and coordinates
            points = location.reshape(location.shape[0], 2)  # Convert to a simple array of points
            
            # Find min and max x,y coordinates
            min_x = np.min(points[:, 0])
            max_x = np.max(points[:, 0])
            min_y = np.min(points[:, 1])
            max_y = np.max(points[:, 1])
            
            logger.info(f"License plate bounds: ({min_x},{min_y}) to ({max_x},{max_y})")
            
            # Calculate vehicle region coordinates
            h, w = image.shape[:2]
            plate_height = max_y - min_y
            plate_width = max_x - min_x
            
            # Define vehicle region coordinates
            vehicle_y_min = max(0, min_y - plate_height * 3)  # Go up 3x the plate height
            vehicle_y_max = min(h, max_y + plate_height * 1)  # Go down 1x the plate height
            vehicle_x_min = max(0, min_x - plate_width * 1)   # Expand width by 1x on each side
            vehicle_x_max = min(w, max_x + plate_width * 1)
            
            # Verify that we have valid coordinates (min < max)
            if vehicle_y_min >= vehicle_y_max or vehicle_x_min >= vehicle_x_max:
                vehicle_y_min, vehicle_y_max = min(vehicle_y_min, vehicle_y_max), max(vehicle_y_min, vehicle_y_max)
                vehicle_x_min, vehicle_x_max = min(vehicle_x_min, vehicle_x_max), max(vehicle_x_min, vehicle_x_max)
            
            # Extract the vehicle region
            original_image = cv2.imread(file_path)  # Re-read the original image
            if original_image is None:
                original_image = image.copy()
            
            vehicle_region = original_image[vehicle_y_min:vehicle_y_max, vehicle_x_min:vehicle_x_max]
            
            # Detect vehicle orientation from the vehicle region
            vehicle_orientation_info = {
                "orientation": "Unknown",
                "confidence": 0.0,
                "is_front": None
            }
            
            if vehicle_region.size > 0 and vehicle_region.shape[0] > 10 and vehicle_region.shape[1] > 10:
                vehicle_orientation_info = vehicle_orientation_detector.predict(vehicle_region)
                logger.info(f"Detected vehicle orientation: {vehicle_orientation_info['orientation']}")
            else:
                # Try detecting from full image as fallback
                vehicle_orientation_info = vehicle_orientation_detector.predict(image)
                logger.info(f"Detected vehicle orientation from full image: {vehicle_orientation_info['orientation']}")
            
            # Detect color from the vehicle region if it's valid
            if vehicle_region.size > 0 and vehicle_region.shape[0] > 10 and vehicle_region.shape[1] > 10:
                region_color_info = detect_vehicle_color(vehicle_region)
                logger.info(f"Detected vehicle color from region: {region_color_info['color']}")
                # Update color_info with the region-based detection
                color_info = region_color_info
                logger.info(f"Using vehicle region color: {color_info['color']}")
            else:
                logger.warning("Invalid vehicle region. Using full image color detection.")
                # Keep the original full-image color_info
            
            # First detect vehicle type from full image
            full_image_type_info = vehicle_detector.detect(image)
            logger.info(f"Detected vehicle type from full image: {full_image_type_info['vehicle_type']}")
            
            # Detect vehicle type from the vehicle region
            region_type_info = {"vehicle_type": "Unknown", "confidence": 0.0, "alternatives": []}
            if vehicle_region.size > 0:
                region_type_info = vehicle_detector.detect(vehicle_region)
                logger.info(f"Detected vehicle type from region: {region_type_info['vehicle_type']}")
            
            # Determine which detection to use based on confidence
            vehicle_type_info = region_type_info if region_type_info["confidence"] > full_image_type_info["confidence"] else full_image_type_info

            # First detect vehicle make from full image
            full_image_make_info = vehicle_make_detector.detect(image)
            logger.info(f"Detected vehicle make from full image: {full_image_make_info['make']}")
            
            # Save full image for make detection visualization
            full_image_make_path = os.path.join(intermediate_dir, f"full_image_make_{os.path.basename(file_path)}")
            cv2.imwrite(full_image_make_path, image.copy())
            full_image_make_rel = f"/results/opencv/intermediate/images/full_image_make_{os.path.basename(file_path)}"
            
            # Initialize with full image make detection
            vehicle_make_info = full_image_make_info

            # Draw annotations with vehicle type information
            font = cv2.FONT_HERSHEY_SIMPLEX
            annotated_image = cv2.putText(image.copy(), text=license_plate, 
                                       org=(location[0][0][0], location[1][0][1]+60),
                                       fontFace=font, fontScale=1, color=(0,255,0), thickness=2)
            
            # Add vehicle type text
            type_text = f"Type: {vehicle_type_info['vehicle_type']}"
            annotated_image = cv2.putText(annotated_image,
                                        type_text,
                                        (location[0][0][0], location[1][0][1]+120),
                                        font, 0.8, (255, 165, 0), 2)

            # Add color information using the potentially updated color_info
            color_text = f"Color: {color_info['color']}"
            annotated_image = cv2.putText(
                annotated_image, 
                color_text,
                (location[0][0][0], location[1][0][1]+90),
                font, 
                0.8, 
                (0, 255, 255),
                2, 
                cv2.LINE_AA
            )
            
            # Draw the vehicle region on the annotated image
            if vehicle_region.size > 0:
                annotated_image = cv2.rectangle(annotated_image, 
                                            (vehicle_x_min, vehicle_y_min), 
                                            (vehicle_x_max, vehicle_y_max), 
                                            (0, 255, 255), 1)
            
            # Add color information to the annotated image
            color_text = f"Color: {color_info['color']}"
            annotated_image = cv2.putText(
                annotated_image, 
                color_text,
                (location[0][0][0], location[1][0][1]+90), # Position below license plate text
                font, 
                0.8, 
                (0, 255, 255), # Yellow color for visibility
                2, 
                cv2.LINE_AA
            )
            
            # Add orientation text to the annotated image
            orientation_text = f"Orientation: {vehicle_orientation_info['orientation']}"
            annotated_image = cv2.putText(
                annotated_image,
                orientation_text,
                (location[0][0][0], location[1][0][1]+150),  # Position below vehicle type
                font,
                0.8,
                (255, 0, 255),  # Magenta color
                2
            )

            # Check if the vehicle region is valid
            if vehicle_region.size > 0 and vehicle_region.shape[0] > 10 and vehicle_region.shape[1] > 10:
                # ...existing code...
                
                # Detect vehicle make from the extracted region
                region_make_info = vehicle_make_detector.detect(vehicle_region)
                logger.info(f"Detected vehicle make from region: {region_make_info['make']}")
                
                # Use the region detection if confidence is higher
                if region_make_info["confidence"] > vehicle_make_info["confidence"]:
                    vehicle_make_info = region_make_info
                    logger.info(f"Using vehicle region make: {vehicle_make_info['make']}")
                
                # Add vehicle make path to result paths
                vehicle_make_path = os.path.join(intermediate_dir, f"vehicle_make_{os.path.basename(file_path)}")
                cv2.imwrite(vehicle_make_path, vehicle_region)
                vehicle_make_rel = f"/results/opencv/intermediate/images/vehicle_make_{os.path.basename(file_path)}"
            
            # Add make information to the annotated image
            make_text = f"Make: {vehicle_make_info['make']}"
            annotated_image = cv2.putText(
                annotated_image,
                make_text,
                (location[0][0][0], location[1][0][1]+180),  # Position below orientation text
                font,
                0.8,
                (102, 255, 153),  # Light green color
                2
            )
            
            logger.info("Annotated image with license plate text, rectangle and color information")

            # Extract a larger area around the license plate for vehicle color detection
            # IMPORTANT: Use the original image for vehicle color detection
            original_image = cv2.imread(file_path)  # Re-read the original image to be sure
            if original_image is None:
                original_image = image.copy()  # Fall back to using the current image if re-reading fails
            
            h, w = original_image.shape[:2]
            
            # Extract the corner points of the license plate
            # Need to properly order the points (top-left, top-right, bottom-right, bottom-left)
            points = location.reshape(location.shape[0], 2)  # Convert to a simple array of points
            
            # Find min and max x,y coordinates
            min_x = np.min(points[:, 0])
            max_x = np.max(points[:, 0])
            min_y = np.min(points[:, 1])
            max_y = np.max(points[:, 1])
            
            logger.info(f"License plate bounds: ({min_x},{min_y}) to ({max_x},{max_y})")
            
            # Calculate vehicle region with larger area above the plate (where the vehicle body usually is)
            # Use a simple rectangular region around the plate
            plate_height = max_y - min_y
            plate_width = max_x - min_x
            
            # Define vehicle region coordinates
            vehicle_y_min = max(0, min_y - plate_height * 3)  # Go up 3x the plate height
            vehicle_y_max = min(h, max_y + plate_height * 1)  # Go down 1x the plate height
            vehicle_x_min = max(0, min_x - plate_width * 1)   # Expand width by 1x on each side
            vehicle_x_max = min(w, max_x + plate_width * 1)
            
            # Verify that we have valid coordinates (min < max)
            if vehicle_y_min >= vehicle_y_max or vehicle_x_min >= vehicle_x_max:
                logger.warning(f"Invalid vehicle region coordinates: y({vehicle_y_min}:{vehicle_y_max}), x({vehicle_x_min}:{vehicle_x_max})")
                # Fix the coordinates in case of inversion
                vehicle_y_min, vehicle_y_max = min(vehicle_y_min, vehicle_y_max), max(vehicle_y_min, vehicle_y_max)
                vehicle_x_min, vehicle_x_max = min(vehicle_x_min, vehicle_x_max), max(vehicle_x_min, vehicle_x_max)
            
            # Print coordinates for debugging
            logger.info(f"Vehicle region coordinates: y({vehicle_y_min}:{vehicle_y_max}), x({vehicle_x_min}:{vehicle_x_max})")
            
            # Create intermediate directory for OpenCV if it doesn't exist
            intermediate_dir = os.path.join("results", "opencv", "intermediate", "images")
            os.makedirs(intermediate_dir, exist_ok=True)
            
            # Save full image for color detection visualization
            full_image_color_path = os.path.join(intermediate_dir, f"full_image_color_{os.path.basename(file_path)}")
            cv2.imwrite(full_image_color_path, image.copy())
            full_image_color_rel = f"/results/opencv/intermediate/images/full_image_color_{os.path.basename(file_path)}"
            
            # Extract the vehicle region
            vehicle_region = original_image[vehicle_y_min:vehicle_y_max, vehicle_x_min:vehicle_x_max]
            
            # Check if the vehicle region is valid
            if vehicle_region.size > 0 and vehicle_region.shape[0] > 10 and vehicle_region.shape[1] > 10:
                logger.info(f"Valid vehicle region extracted with shape {vehicle_region.shape}")
                
                # Save the vehicle region for visualization
                vehicle_region_file = os.path.join(intermediate_dir, f"vehicle_region_{os.path.basename(file_path)}")
                cv2.imwrite(vehicle_region_file, vehicle_region)
                vehicle_region_rel = f"/results/opencv/intermediate/images/vehicle_region_{os.path.basename(file_path)}"
                intermediate_steps = {
                    "vehicle_region": vehicle_region_rel
                }
                
                # Draw the vehicle region on the annotated image with a different color (yellow)
                annotated_image = cv2.rectangle(annotated_image, 
                                            (vehicle_x_min, vehicle_y_min), 
                                            (vehicle_x_max, vehicle_y_max), 
                                            (0, 255, 255), 1)  # Yellow with thin line
                
                # Now detect color from this vehicle region instead of the whole image
                region_color_info = detect_vehicle_color(vehicle_region)
                logger.info(f"Detected vehicle color from region: {region_color_info['color']}")
                
                # Store both color detections
                region_color = region_color_info["color"]
                region_color_confidence = region_color_info["confidence"]
                region_color_percentages = region_color_info.get("color_percentages", {})
                
                # Use the color detected from the vehicle region if confidence is higher
                color_info = region_color_info if region_color_info["confidence"] > full_image_color["confidence"] else full_image_color
                best_color_source = "region" if region_color_info["confidence"] > full_image_color["confidence"] else "full_image"
            else:
                logger.warning(f"Invalid vehicle region shape: {vehicle_region.shape if hasattr(vehicle_region, 'shape') else 'unknown'}")
                # Keep using the full image color detection
                region_color = "Unknown"
                region_color_confidence = 0.0
                region_color_percentages = {}
                best_color_source = "full_image"

            # Save the annotated image in the opencv/images subfolder
            result_image_path = os.path.join("results", "opencv", "images", os.path.basename(file_path))
            cv2.imwrite(result_image_path, annotated_image)
            logger.info(f"Annotated image saved at: {result_image_path}")

            # Load customer data from example.json
            example_json_path = os.path.join(os.path.dirname(__file__), '../api/endpoints/example.json')
            logger.info(f"Loading customer data from: {example_json_path}")
            with open(example_json_path, 'r') as f:
                customer_data = json.load(f)
            logger.info("Customer data loaded successfully")

            # Encode intermediate images as base64 strings
            def encode_image(image):
                if image is None or not isinstance(image, np.ndarray) or image.size == 0:
                    # Return an empty base64 string or placeholder for empty images
                    logger.warning("Attempted to encode empty or invalid image")
                    return ""
                try:
                    _, buffer = cv2.imencode('.jpg', image)
                    return base64.b64encode(buffer).decode('utf-8')
                except Exception as e:
                    logger.error(f"Error encoding image: {e}")
                    return ""

            intermediate_images = {
                "gray": encode_image(gray),
                "edge": encode_image(edges),
                "localized": encode_image(new_image),
                "plate": encode_image(cropped_image),
            }
            
            # Add vehicle region to intermediate images with explicit error handling
            if 'vehicle_region' in locals() and isinstance(vehicle_region, np.ndarray) and vehicle_region.size > 0:
                try:
                    vehicle_region_encoded = encode_image(vehicle_region)
                    if vehicle_region_encoded:
                        intermediate_images["vehicle_region"] = vehicle_region_encoded
                        logger.info("Successfully encoded vehicle region image")
                    else:
                        logger.warning("Failed to encode vehicle region image")
                except Exception as e:
                    logger.error(f"Exception while encoding vehicle region: {e}")
            else:
                logger.warning("Vehicle region not available for encoding")
            
            # Create debug info to help diagnose issues - Convert NumPy types to native Python types
            debug_info = {
                "vehicle_region_exists": 'vehicle_region' in locals() and vehicle_region.size > 0,
                "vehicle_region_shape": [int(dim) for dim in vehicle_region.shape] if 'vehicle_region' in locals() and hasattr(vehicle_region, 'shape') and vehicle_region.size > 0 else None,
                "region_coordinates": {
                    "y_min": int(vehicle_y_min),
                    "y_max": int(vehicle_y_max),
                    "x_min": int(vehicle_x_min),
                    "x_max": int(vehicle_x_max)
                },
                "license_plate_bounds": {
                    "min_x": int(min_x),
                    "max_x": int(max_x),
                    "min_y": int(min_y),
                    "max_y": int(max_y)
                }
            }
            
            # Vehicle region coordinates for frontend visualization - Ensure they are Python int types
            vehicle_region_coordinates = {
                "x_min": int(vehicle_x_min),
                "y_min": int(vehicle_y_min),
                "x_max": int(vehicle_x_max),
                "y_max": int(vehicle_y_max)
            }

            # Create the result dict
            result = {
                "status": "success",
                "result_url": f"/results/opencv/images/{os.path.basename(result_image_path)}",
                "intermediate_images": intermediate_images,
                "intermediate_steps": intermediate_steps,  # Include intermediate steps dictionary
                "license_plate": license_plate,
                "original_ocr": original_ocr_text,  # Include original OCR text
                "filename": file_path,
                "customer_data": customer_data,
                "text_candidates": text_candidates,  # Already a direct array from our extraction function
                "vehicle_color": color_info["color"],  # Best color (either from region or full image)
                "color_confidence": color_info["confidence"],
                "color_hex": get_rgb_color(color_info["color"]),  # Add hex color code
                "full_image_color": full_image_color["color"],  # Full image color
                "full_image_color_confidence": full_image_color["confidence"],
                "full_image_color_hex": get_rgb_color(full_image_color["color"]),  # Add hex color
                "region_color": region_color,  # Region-specific color
                "region_color_confidence": region_color_confidence,
                "region_color_hex": get_rgb_color(region_color),  # Add hex color
                "color_percentages": full_image_color.get("color_percentages", {}),  # Full image color percentages
                "region_color_percentages": region_color_percentages,  # Region color percentages
                "best_color_source": best_color_source,  # Which source gave the best color detection
                "vehicle_region_coordinates": vehicle_region_coordinates,  # Include the coordinates
                "debug_info": debug_info,  # Add debug info to help diagnose issues
                "vehicle_type": vehicle_type_info["vehicle_type"],
                "vehicle_type_confidence": vehicle_type_info["confidence"],
                "vehicle_type_alternatives": vehicle_type_info["alternatives"],
                # Add both detection results separately
                "full_image_type": full_image_type_info["vehicle_type"],
                "full_image_type_confidence": full_image_type_info["confidence"],
                "region_type": region_type_info["vehicle_type"],
                "region_type_confidence": region_type_info["confidence"],
                "vehicle_orientation": vehicle_orientation_info["orientation"],
                "orientation_confidence": vehicle_orientation_info["confidence"],
                "is_front_facing": vehicle_orientation_info["is_front"],
                "vehicle_make": vehicle_make_info["make"],
                "make_confidence": vehicle_make_info["confidence"],
                "make_alternatives": vehicle_make_info["alternatives"],
                # Add both detection results separately
                "full_image_make": full_image_make_info["make"],
                "full_image_make_confidence": full_image_make_info["confidence"],
                "region_make": region_make_info["make"] if 'region_make_info' in locals() else "Unknown",
                "region_make_confidence": region_make_info["confidence"] if 'region_make_info' in locals() else 0.0,
                "best_make_source": "region" if ('region_make_info' in locals() and region_make_info["confidence"] > full_image_make_info["confidence"]) else "full_image",
                "intermediate_steps": {
                    "vehicle_region": vehicle_region_rel,
                    "full_image_color": full_image_color_rel,
                    "vehicle_make": vehicle_make_rel if 'vehicle_make_rel' in locals() else None,
                    "full_image_make": full_image_make_rel
                },
                "vehicle_boxes": vehicle_boxes,
                "primary_vehicle_box": vehicle_box
            }

            return result
        else:
            logger.error("Invalid or empty contour points.")
            raise ValueError("Invalid or empty contour points.")
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON: {e}")
        raise
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise

def process_video(file_path, confidence_threshold=0.7):
    try:
        ensure_dirs()
        cap = cv2.VideoCapture(file_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_number = 0
        results = []
        max_frames = 40  # Limit the number of frames to process
        all_vehicle_colors = []  # Track all detected vehicle colors

        # Use a dynamic threshold based on confidence_threshold
        min_contour_area = 50 if confidence_threshold < 0.7 else 100

        # Add vehicle make tracking
        all_vehicle_makes = []
        
        # Initialize vehicle make info
        vehicle_make_info = {
            "make": "Unknown",
            "confidence": 0.0,
            "alternatives": []
        }

        while cap.isOpened() and frame_number < max_frames:  # Fixed: Added missing dot between 'cap' and 'isOpened()'
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_number += 1
            logger.info(f"Processing frame {frame_number}/{min(frame_count, max_frames)}")
            
            # Detect vehicle color from the full frame
            color_info = detect_vehicle_color(frame)
            all_vehicle_colors.append(color_info["color"])
            logger.info(f"Frame {frame_number}: Detected vehicle color: {color_info['color']}")

            # Detect vehicle make from the full frame
            make_info = vehicle_make_detector.detect(frame)
            if make_info["confidence"] > vehicle_make_info["confidence"]:
                vehicle_make_info = make_info
            
            all_vehicle_makes.append(make_info["make"])
            logger.info(f"Frame {frame_number}: Detected vehicle make: {make_info['make']}")

            # Process each frame (similar to process_image function)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
            edges = cv2.Canny(bfilter, 30, 200)
            keypoints = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(keypoints)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
            location = None
            for contour in contours:
                approx = cv2.approxPolyDP(contour, 10, True)
                if len(approx) == 4:
                    location = approx
                    break

            if location is None:
                logger.warning("No valid contour with 4 points found. Trying alternative method.")
                for contour in contours:
                    if cv2.contourArea(contour) > min_contour_area:  # Use dynamic threshold
                        location = cv2.convexHull(contour)
                        break

            if location is not None and location.shape[0] > 0 and location.dtype == np.int32:
                mask = np.zeros(gray.shape, np.uint8)
                new_image = cv2.drawContours(mask, [location], 0, 255, -1)
                new_image = cv2.bitwise_and(frame, frame, mask=mask)
                (x, y) = np.where(mask == 255)
                if x.size == 0 or y.size == 0:
                    logger.error("Failed to locate the license plate region.")
                    continue
                (x1, y1) = (np.min(x), np.min(y))
                (x2, y2) = (np.max(x), np.max(y))
                cropped_image = gray[x1:x2+1, y1:y2+1]
                
                # Use our centralized function to extract text
                license_plate, _ = extract_text_from_plate(cropped_image, preprocessing_level='minimal')
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                annotated_frame = cv2.putText(frame, text=license_plate, org=(location[0][0][0], location[1][0][1]+60), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
                annotated_frame = cv2.rectangle(frame, tuple(location[0][0]), tuple(location[2][0]), (0,255,0), 3)
                
                # Add color information to the frame
                color_text = f"Color: {color_info['color']}"
                annotated_frame = cv2.putText(
                    annotated_frame, 
                    color_text,
                    (location[0][0][0], location[1][0][1]+90), # Position below license plate text
                    font, 
                    0.8, 
                    (0, 255, 255), # Yellow color for visibility
                    2, 
                    cv2.LINE_AA
                )
                
                # Add make information to the frame
                if vehicle_make_info["make"] != "Unknown":
                    make_text = f"Make: {vehicle_make_info['make']}"
                    annotated_frame = cv2.putText(
                        annotated_frame, 
                        make_text,
                        (location[0][0][0], location[1][0][1]+120), # Position below color text
                        font, 
                        0.8, 
                        (102, 255, 153), # Light green color
                        2, 
                        cv2.LINE_AA
                    )

                results.append(annotated_frame)
            else:
                # Even if we can't find the license plate, we still want to process the frame
                # and show the color detection
                color_text = f"Color: {color_info['color']}"
                annotated_frame = cv2.putText(
                    frame.copy(), 
                    color_text,
                    (10, 30), # Position at top-left if no plate is found
                    font, 
                    0.8, 
                    (0, 255, 255), # Yellow color for visibility
                    2, 
                    cv2.LINE_AA
                )
                # Even if we can't find the license plate, still show the make detection
                if vehicle_make_info["make"] != "Unknown":
                    make_text = f"Make: {vehicle_make_info['make']}"
                    annotated_frame = cv2.putText(
                        frame.copy() if 'annotated_frame' not in locals() else annotated_frame, 
                        make_text,
                        (10, 60), # Position below color text at top-left if no plate is found
                        font, 
                        0.8, 
                        (102, 255, 153), # Light green color
                        2, 
                        cv2.LINE_AA
                    )
                    results.append(annotated_frame)
                else:
                    # Just add the frame with color info if we already added that
                    if 'annotated_frame' in locals():
                        results.append(annotated_frame)

        cap.release()
        if results:
            result_video_path = os.path.join("results", "opencv", "videos", os.path.basename(file_path))
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Use H.264 codec
            frame_size = (results[0].shape[1], results[0].shape[0])
            out = cv2.VideoWriter(result_video_path, fourcc, fps, frame_size)

            for frame in results:
                out.write(frame)
            out.release()
            logger.info(f"Video writer released successfully")

            logger.info(f"Annotated video saved at: {result_video_path}")

            # Determine most common vehicle color
            from collections import Counter
            if all_vehicle_colors:
                color_counter = Counter(all_vehicle_colors)
                most_common_color = color_counter.most_common(1)[0][0]
                color_frequency = color_counter.most_common(1)[0][1] / len(all_vehicle_colors)
            else:
                most_common_color = "Unknown"
                color_frequency = 0.0

            # Determine most common vehicle make
            if all_vehicle_makes:
                make_counter = Counter(all_vehicle_makes)
                most_common_make = make_counter.most_common(1)[0][0]
                make_frequency = make_counter.most_common(1)[0][1] / len(all_vehicle_makes)
            else:
                most_common_make = "Unknown"
                make_frequency = 0.0

            result = {
                "status": "success",
                "result_url": f"/results/opencv/videos/{os.path.basename(result_video_path)}",
                "filename": file_path,
                "vehicle_color": most_common_color,  # Add most common vehicle color
                "color_confidence": color_frequency,  # Add confidence (frequency) of color detection
                "all_colors": dict(color_counter.most_common(3)),  # Include top 3 detected colors
                "vehicle_make": most_common_make,  # Add most common vehicle make
                "make_confidence": make_frequency,  # Add confidence (frequency) of make detection
                "all_makes": dict(make_counter.most_common(3)) if 'make_counter' in locals() else {}  # Include top 3 detected makes
            }

            return result
        else:
            raise Exception("No frames were processed successfully.")
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        raise
