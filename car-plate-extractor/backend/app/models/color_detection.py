import cv2
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_vehicle_color(image, vehicle_box=None):
    """
    Detect the dominant color of a vehicle in an image.
    
    Args:
        image: The image containing the vehicle
        vehicle_box: Optional bounding box of vehicle (x_min, y_min, x_max, y_max)
                     If provided, only this region will be used for color detection
    
    Returns:
        Dictionary with the color name, confidence, and color percentages.
    """
    try:
        if image is None or image.size == 0:
            logger.error("Invalid image provided to color detection")
            return {
                "color": "Unknown",
                "confidence": 0.0,
                "color_percentages": {}
            }
        
        # If vehicle_box is provided, extract just that region
        if vehicle_box is not None:
            x_min, y_min, x_max, y_max = vehicle_box
            if x_min < x_max and y_min < y_max:
                # Ensure coordinates are within image bounds
                h, w = image.shape[:2]
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(w, x_max)
                y_max = min(h, y_max)
                
                # Extract vehicle region
                if x_max > x_min and y_max > y_min:
                    image = image[y_min:y_max, x_min:x_max]
                    logger.info(f"Using vehicle region for color detection: {x_min},{y_min} to {x_max},{y_max}")
        
        # Convert to HSV for better color discrimination
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Improved color ranges in HSV - carefully calibrated for vehicle colors
        color_ranges = {
            # White: Very high value, very low saturation
            "white": [(0, 0, 190), (180, 30, 255)],
            
            # Black: Very low value
            "black": [(0, 0, 0), (180, 50, 40)],
            
            # Gray: Low saturation, medium value
            "gray": [(0, 0, 40), (180, 30, 190)],
            
            # Silver: Low saturation, high value (between gray and white)
            "silver": [(0, 0, 170), (180, 30, 230)],
            
            # Red spans across the color wheel (near 0 and 180)
            "red1": [(0, 60, 50), (10, 255, 255)],
            "red2": [(170, 60, 50), (180, 255, 255)],
            
            # Orange: Between red and yellow
            "orange": [(11, 100, 100), (20, 255, 255)],
            
            # Yellow: Narrow range to avoid confusion with gold/beige
            "yellow": [(21, 100, 100), (35, 255, 255)],
            
            # Green: Wide range to capture different greens
            "green": [(36, 50, 50), (85, 255, 255)],
            
            # Blue: Clear sky blue to deep navy
            "blue": [(86, 50, 50), (130, 255, 255)],
            
            # Purple: Between blue and red
            "purple": [(131, 50, 50), (160, 255, 255)],
            
            # Brown: Low saturation reddish tones
            "brown": [(5, 50, 50), (18, 150, 150)]
        }
        
        # Create masks for each color
        color_counts = {}
        color_masks = {}
        
        # Get total number of pixels
        total_pixels = image.shape[0] * image.shape[1]
        
        # First pass: compute raw color percentages (except red which needs special handling)
        for color, ranges in color_ranges.items():
            if color == "red1" or color == "red2":
                # Skip red1/red2 in first pass, we'll combine them later
                continue
                
            if len(ranges) == 2:
                lower, upper = ranges
                mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
                color_masks[color] = mask
                color_counts[color] = cv2.countNonZero(mask)
        
        # Special case for red (combines red1 and red2 ranges)
        lower1, upper1 = color_ranges["red1"]
        lower2, upper2 = color_ranges["red2"]
        mask1 = cv2.inRange(hsv_image, np.array(lower1), np.array(upper1))
        mask2 = cv2.inRange(hsv_image, np.array(lower2), np.array(upper2))
        red_mask = cv2.bitwise_or(mask1, mask2)
        color_masks["red"] = red_mask
        color_counts["red"] = cv2.countNonZero(red_mask)
        
        # Calculate brightness and saturation stats
        v_channel = hsv_image[:,:,2]
        s_channel = hsv_image[:,:,1]
        avg_v = np.mean(v_channel)
        avg_s = np.mean(s_channel)
        
        # Count pixels in dark and light regions
        dark_mask = v_channel < 50
        light_mask = v_channel > 190
        dark_ratio = np.sum(dark_mask) / total_pixels
        light_ratio = np.sum(light_mask) / total_pixels
        
        # Count low saturation pixels
        low_sat_mask = s_channel < 40
        low_sat_ratio = np.sum(low_sat_mask) / total_pixels
        
        logger.info(f"Brightness stats - avg_v: {avg_v}, dark ratio: {dark_ratio:.2f}, light ratio: {light_ratio:.2f}")
        logger.info(f"Saturation stats - avg_s: {avg_s}, low sat ratio: {low_sat_ratio:.2f}")
        
        # Refine achromatic colors (black, white, gray, silver) using statistics
        
        # For very dark images, boost black
        if avg_v < 60 and dark_ratio > 0.4:
            color_counts["black"] = int(total_pixels * 0.9)
            for color in color_counts:
                if color != "black":
                    color_counts[color] = int(color_counts.get(color, 0) * 0.2)
        
        # For very bright images with low saturation, boost white
        elif avg_v > 180 and light_ratio > 0.4 and low_sat_ratio > 0.7:
            color_counts["white"] = int(total_pixels * 0.9)
            for color in color_counts:
                if color != "white":
                    color_counts[color] = int(color_counts.get(color, 0) * 0.2)
        
        # For medium brightness with low saturation, might be gray or silver
        elif 70 <= avg_v <= 160 and low_sat_ratio > 0.7:
            if avg_v < 130:
                color_counts["gray"] = int(total_pixels * 0.8)
            else:
                color_counts["silver"] = int(total_pixels * 0.8)
            
            # Reduce other colors
            for color in color_counts:
                if color != "gray" and color != "silver":
                    color_counts[color] = int(color_counts.get(color, 0) * 0.2)
        
        # Calculate color percentages
        color_percentages = {}
        total_count = sum(color_counts.values())
        for color, count in color_counts.items():
            if total_count > 0:
                percentage = (count / total_count) * 100
            else:
                percentage = 0
            color_percentages[color] = round(percentage, 1)
        
        # Sort colors by percentage
        sorted_colors = sorted(color_percentages.items(), key=lambda x: x[1], reverse=True)
        
        # Get dominant color
        dominant_color = sorted_colors[0][0] if sorted_colors else "Unknown"
        dominant_percentage = sorted_colors[0][1] if sorted_colors else 0
        
        # Calculate confidence based on how dominant the color is
        if len(sorted_colors) > 1:
            second_percentage = sorted_colors[1][1]
            margin = dominant_percentage - second_percentage
            
            # Higher margin = higher confidence
            # Scale confidence from 0.5-1.0 based on margin
            # A margin of 30 points or more gives full confidence
            confidence = min(1.0, 0.5 + (margin / 60))
            
            # Higher confidence for achromatic colors when statistics strongly support it
            if dominant_color == "black" and dark_ratio > 0.5:
                confidence = max(confidence, 0.9)
            elif dominant_color == "white" and light_ratio > 0.5 and low_sat_ratio > 0.7:
                confidence = max(confidence, 0.9)
            elif dominant_color == "gray" and low_sat_ratio > 0.7 and 70 <= avg_v <= 130:
                confidence = max(confidence, 0.8)
            elif dominant_color == "silver" and low_sat_ratio > 0.7 and 130 < avg_v <= 170:
                confidence = max(confidence, 0.8)
        else:
            confidence = 0.8  # Default confidence if only one color detected
        
        # Log detected color information
        logger.info(f"Detected color: {dominant_color} with confidence {confidence:.2f}")
        
        # Print top 3 color percentages for debugging
        top_colors = sorted_colors[:3]
        color_debug = ", ".join([f"{color}: {percentage:.1f}%" for color, percentage in top_colors])
        logger.info(f"Color percentages: {color_debug}")
        
        return {
            "color": dominant_color,
            "confidence": confidence,
            "color_percentages": dict(color_percentages)
        }
    except Exception as e:
        logger.error(f"Error in color detection: {e}")
        return {
            "color": "Unknown",
            "confidence": 0.0,
            "color_percentages": {}
        }

def visualize_color_detection(image, vehicle_box=None):
    """
    Visualize the color detection process by creating an image with the detected color
    patches and information.
    """
    try:
        if image is None or image.size == 0:
            logger.error("Invalid image provided to color visualization")
            return None
            
        # Make a copy of the image to avoid modifying the original
        viz_image = image.copy()
        
        # Highlight the vehicle region if provided
        if vehicle_box is not None:
            x_min, y_min, x_max, y_max = vehicle_box
            cv2.rectangle(viz_image, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
        
        # Detect color
        color_info = detect_vehicle_color(image, vehicle_box)
        
        # Create visualization image
        height, width = image.shape[:2]
        vis_height = max(height, 300)
        visualization = np.zeros((vis_height, width + 250, 3), dtype=np.uint8)
        
        # Copy original image to left side
        visualization[:height, :width] = viz_image
        
        # Add color patches and labels on right side
        color_percentages = color_info.get("color_percentages", {})
        sorted_colors = sorted(color_percentages.items(), key=lambda x: x[1], reverse=True)
        
        # Draw top 5 colors
        for i, (color_name, percentage) in enumerate(sorted_colors[:5]):
            # Create color patch
            y_start = 50 + i * 50
            patch = np.ones((40, 40, 3), dtype=np.uint8) * get_bgr_color(color_name)
            
            # Place patch in visualization
            if y_start + 40 <= vis_height:
                visualization[y_start:y_start+40, width+20:width+60] = patch
                
                # Add color name and percentage
                text = f"{color_name}: {percentage:.1f}%"
                cv2.putText(visualization, text, (width+70, y_start+25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Add dominant color label
        cv2.putText(visualization, f"Dominant: {color_info['color']}", 
                   (width+20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add confidence
        conf_text = f"Confidence: {color_info['confidence']:.2f}"
        cv2.putText(visualization, conf_text, 
                   (width+20, vis_height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return visualization
        
    except Exception as e:
        logger.error(f"Error in color visualization: {e}")
        return None

def get_bgr_color(color_name):
    """Map color name to BGR value for visualization"""
    color_map = {
        "red": (0, 0, 255),       # BGR for pure red
        "green": (0, 255, 0),     # BGR for pure green
        "blue": (255, 0, 0),      # BGR for pure blue
        "yellow": (0, 255, 255),  # BGR for yellow (green + red)
        "orange": (0, 165, 255),  # BGR for orange
        "purple": (255, 0, 255),  # BGR for purple (magenta)
        "white": (255, 255, 255), # BGR for white
        "black": (0, 0, 0),       # BGR for black
        "gray": (128, 128, 128),  # BGR for medium gray
        "silver": (192, 192, 192),# BGR for silver (light gray)
        "brown": (42, 42, 165)    # BGR for brown
    }
    return color_map.get(color_name, (200, 200, 200))  # Default to light gray

def get_rgb_color(color_name):
    """Map color name to RGB value for frontend visualization
    Returns a hex string in #RRGGBB format
    """
    # Get BGR color first
    bgr_color = get_bgr_color(color_name)
    
    # Convert BGR to RGB
    rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])
    
    # Convert to hex string
    hex_color = "#{:02x}{:02x}{:02x}".format(rgb_color[0], rgb_color[1], rgb_color[2])
    
    return hex_color
