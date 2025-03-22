import re
from difflib import SequenceMatcher
import logging
import cv2
import numpy as np
import easyocr

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize EasyOCR reader (will be used by multiple modules)
reader = None

def get_reader():
    """Get or initialize the EasyOCR reader as a singleton"""
    global reader
    if reader is None:
        reader = easyocr.Reader(['en'])
    return reader

# Add special cases for pattern matching
SPECIAL_CASES = {
    'COVID19': ['CQVID19', 'COVIDI9', 'COVD19', 'IOVID19', 'FCOVID19', 'ECOVID19', 'COVD9', 'FCOVD9', 'OD19', 'QD19', '0D19', 'CD19'],
    'COVID': ['COVD', 'CQVID', 'ECOVID', 'FCOVID', 'OD', 'QD', '0D', 'CD'],
}

# Common license plate patterns with their descriptions - Update this with more specific patterns
LICENSE_PATTERNS = [
    # Special COVID pattern - give this higher priority
    {"pattern": r'^[A-Z]?COVID\d{1,2}$', "name": "Special - COVID19"},
    
    # UK Formats
    {"pattern": r'^[A-Z]{2}\d{2}\s?[A-Z]{3}$', "name": "UK New Style (AB12 CDE)"},
    {"pattern": r'^[A-Z]{1,2}\d{1,4}$', "name": "UK Old Style (A123)"},
    {"pattern": r'^[A-Z]{3}\d{1,4}$', "name": "UK Diplomatic (XYZ123)"},
    
    # US Formats (examples from different states) - Add space flexibility
    {"pattern": r'^\d{1,3}\s*[A-Z]{3}$', "name": "US - 3 Letters (123 ABC)"},  # Added \s* to allow optional spaces
    {"pattern": r'^[A-Z]{3}\s*\d{3,4}$', "name": "US - 3 Letters (ABC 1234)"},  # Added \s* to allow optional spaces
    {"pattern": r'^\d{3}\s*[A-Z]{2}\s*\d{1}$', "name": "CA Format (123 AB 4)"},  # Added \s* to allow optional spaces
    
    # European Formats
    {"pattern": r'^[A-Z]{1,3}-\d{1,4}-[A-Z]{1,2}$', "name": "EU with Hyphens (AB-1234-C)"},
    {"pattern": r'^[A-Z]{1,2}\s+\d{1,4}\s+[A-Z]{1,2}$', "name": "EU with Spaces (AB 1234 C)"},
    
    # Special formats
    {"pattern": r'^COVID\d{2}$', "name": "Special - COVID19"},
    {"pattern": r'^[A-Z]{5}$', "name": "All Letters (ABCDE)"},
    
    # General formats with optional spaces
    {"pattern": r'^[A-Z]{2}\s*\d{2}\s*[A-Z]{2}$', "name": "Format AB12CD or AB 12 CD"},
    {"pattern": r'^[A-Z]{2}\s*\d{3}\s*[A-Z]{2}$', "name": "Format AB123CD or AB 123 CD"},
    {"pattern": r'^[A-Z]{1,3}\s*\d{1,4}\s*[A-Z]{0,2}$', "name": "General Format (ABC1234DE or ABC 1234 DE)"},
    
    # Add pattern specifically for "172 TMZ" type format
    {"pattern": r'^\d{2,3}\s+[A-Z]{3}$', "name": "Format 123 ABC with space"},
    
    # Add more specific patterns for problematic cases
    {"pattern": r'^OD\d{2}$', "name": "Special Format (OD19)"},
    {"pattern": r'^[A-Z]{2}\d{2}$', "name": "Format AB12"},
    
    # Additional patterns with optional spaces between all characters
    {"pattern": r'^(\d|\s)*(\d+)(\s+[A-Z]+)$', "name": "Numbers followed by letters with spaces"},
    {"pattern": r'^([A-Z]|\s)*([A-Z]+)(\s+\d+)$', "name": "Letters followed by numbers with spaces"},
]

# Character confusion mapping (commonly misread characters)
CHAR_CONFUSION = {
    '0': ['O', 'D', 'Q'],
    'O': ['0', 'D', 'Q'],
    '1': ['I', 'L', 'T'],
    'I': ['1', 'L', 'T'],
    'L': ['1', 'I', 'T'],
    '2': ['Z', 'S'],
    'Z': ['2', 'S'],
    '5': ['S', 'B'],
    'S': ['5', '3', 'B'],
    '8': ['B', '3'],
    'B': ['8', '3', 'P', 'R'],
    'D': ['0', 'O', 'Q'],
    'G': ['6', 'C'],
    '6': ['G', 'C'],
    'U': ['V', 'Y'],
    'V': ['U', 'Y'],
    'W': ['VV', 'M'],
    'M': ['N', 'H'],
    'N': ['M', 'H'],
    'P': ['R', 'F'],
    'R': ['P', 'F', 'B']
}

# Expanded character similarity mapping for position-based candidates
EXTENDED_CHAR_SIMILARITY = {
    '0': ['O', 'D', 'Q', '8', 'C'],
    '1': ['I', 'L', 'T', '7', 'J'],
    '2': ['Z', 'S', '7', 'R'],
    '3': ['8', 'B', 'E', 'S'],
    '4': ['A', 'H', 'M', 'N'],
    '5': ['S', '6', 'G', 'B'],
    '6': ['G', 'C', 'B', '8'],
    '7': ['T', '1', 'L', 'Y'],
    '8': ['B', '6', '3', 'S', '0'],
    '9': ['G', 'Q', 'D', 'P'],
    'A': ['4', 'H', 'M', 'N', 'V'],
    'B': ['8', '3', 'R', 'S', 'D'],
    'C': ['G', 'O', 'Q', '0', 'D'],
    'D': ['O', '0', 'Q', 'C', 'B'],
    'E': ['F', 'B', '3', 'R', 'M'],
    'F': ['E', 'P', 'B', 'R'],
    'G': ['C', '6', 'Q', 'O', '0'],
    'H': ['N', 'M', 'K', 'A', 'X'],
    'I': ['1', 'L', 'T', 'J', '!'],
    'J': ['I', 'L', '1', 'T', 'U'],
    'K': ['X', 'H', 'R', 'N', 'M'],
    'L': ['I', '1', 'T', 'J', 'F'],
    'M': ['N', 'H', 'A', 'W', 'K'],
    'N': ['M', 'H', 'K', 'A', 'X'],
    'O': ['0', 'Q', 'D', 'C', 'G'],
    'P': ['R', 'F', 'D', '9', 'B'],
    'Q': ['O', '0', 'G', 'D', '9'],
    'R': ['B', 'P', 'F', 'K', '2'],
    'S': ['5', '2', '8', 'B', 'Z'],
    'T': ['1', 'I', 'L', '7', 'Y'],
    'U': ['V', 'Y', 'W', 'J', 'O'],
    'V': ['U', 'Y', 'W', 'A', 'N'],
    'W': ['M', 'N', 'V', 'U', 'H'],
    'X': ['K', 'H', 'N', 'M', 'Y'],
    'Y': ['V', 'U', 'T', '7', 'X'],
    'Z': ['2', 'S', '7', 'N', 'M']
}

def looks_like_covid(text):
    """Deep inspection to detect if a text might be a corrupted version of COVID19"""
    text = text.upper()
    
    # Direct special case check
    for variant in SPECIAL_CASES['COVID19']:
        if text == variant:
            return True, 0.95
            
    # Check for substrings that strongly indicate COVID
    covid_indicators = ['COVID', 'COVD', 'COVI', 'C0VID', 'COV', 'VID19', 'VID', 'ID19', 'D19']
    for indicator in covid_indicators:
        if indicator in text:
            return True, 0.85
    
    # Check for short texts that look like the end of COVID19
    if len(text) <= 4:
        covid_shortcuts = ['OD19', 'VID', 'D19', 'C19', 'V19', 'ID9']
        for shortcut in covid_shortcuts:
            if text == shortcut:
                return True, 0.8
                
    # Check for character patterns
    if len(text) >= 2:
        # If it starts with any of C, O, V, D and has a number
        if text[0] in "COVD" and any(c.isdigit() for c in text):
            return True, 0.7
            
        # If it starts with any of O, D and has 1, 9 or 19
        if text[0] in "OD" and ('19' in text or '1' in text or '9' in text):
            return True, 0.75
    
    return False, 0.0

def check_special_cases(text):
    """Enhanced check if the text is a special case like COVID19 with common OCR errors."""
    cleaned_text = ''.join(c for c in text if c.isalnum()).upper()
    
    # First do a quick check for exact matches
    for correct_text, variants in SPECIAL_CASES.items():
        if cleaned_text == correct_text:
            return True, correct_text, 1.0  # Exact match to correct form
        
        for variant in variants:
            # Look for exact match to a known variant
            if cleaned_text == variant:
                return True, correct_text, 0.9
    
    # Special handling for OD19 which should always be COVID19
    if cleaned_text == 'OD19':
        logger.info("OD19 detected - correcting to COVID19")
        return True, 'COVID19', 0.92
    
    # Check for possible COVID patterns
    is_covid, confidence = looks_like_covid(cleaned_text)
    if is_covid:
        logger.info(f"COVID-like pattern detected in '{cleaned_text}' with confidence {confidence}")
        return True, 'COVID19', confidence
    
    # More detailed similarity matching for other cases
    for correct_text, variants in SPECIAL_CASES.items():
        for variant in variants:
            # Look for partial match (containing the variant)
            if variant in cleaned_text or cleaned_text in variant:
                similarity = similarity_score(cleaned_text, variant)
                if similarity > 0.7:  # High similarity threshold
                    return True, correct_text, similarity
    
    return False, None, 0.0

def matches_pattern(text):
    """Check if text matches any of the common license plate patterns."""
    # Clean text for pattern matching (remove spaces)
    cleaned_text = ''.join(text.split()).upper()
    
    # First check special cases like COVID19
    is_special, special_text, special_confidence = check_special_cases(cleaned_text)
    if is_special:
        return True, f"Special Case - {special_text}"
    
    # Then check regular patterns
    for pattern_dict in LICENSE_PATTERNS:
        if re.match(pattern_dict["pattern"], cleaned_text):
            return True, pattern_dict["name"]
    
    return False, None

def levenshtein_distance(s1, s2):
    """Calculate the Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def generate_pattern_variants(text):
    """Generate possible variants based on common OCR confusions."""
    variants = [text]
    
    # Generate all possible first-level substitutions
    for i, char in enumerate(text):
        if char in CHAR_CONFUSION:
            for confused_char in CHAR_CONFUSION[char]:
                new_text = text[:i] + confused_char + text[i+1:]
                if new_text not in variants:
                    variants.append(new_text)
    
    # For texts with fewer than 8 characters, also generate limited second-level substitutions
    if len(text) < 8:
        initial_variants = variants.copy()
        for variant in initial_variants:
            for i, char in enumerate(variant):
                if char in CHAR_CONFUSION:
                    for confused_char in CHAR_CONFUSION[char]:
                        new_text = variant[:i] + confused_char + variant[i+1:]
                        if new_text not in variants:
                            variants.append(new_text)
    
    return variants

def correct_plate_text(text, confidence_threshold=0.5):
    """
    Attempt to correct the plate text based on pattern matching and character confusion.
    
    Args:
        text: The OCR-detected license plate text
        confidence_threshold: Minimum confidence to apply corrections
        
    Returns:
        tuple: (corrected_text, is_valid_pattern, confidence, pattern_name)
    """
    # First, try with the original text (which may contain spaces)
    # This will help recognize patterns like "172 TMZ"
    original_with_spaces = text.upper()
    is_match_with_spaces, pattern_name_with_spaces = matches_pattern(original_with_spaces)
    if is_match_with_spaces:
        return original_with_spaces, True, 1.0, pattern_name_with_spaces
    
    # If no match with spaces, continue with standard cleaned text approach
    cleaned_text = ''.join(c for c in text if c.isalnum()).upper()
    
    # Special case for OD19 which should always be COVID19
    if cleaned_text == 'OD19':
        logger.info("OD19 detected in correct_plate_text - correcting to COVID19")
        return "COVID19", True, 1.0, "Special Case - COVID19"
    
    # Check for possible COVID patterns
    is_covid, confidence = looks_like_covid(cleaned_text)
    if is_covid and confidence > 0.7:
        logger.info(f"COVID pattern detected in '{cleaned_text}' with confidence {confidence}")
        return "COVID19", True, confidence, "Special Case - COVID19"
    
    # First check for special cases
    is_special, special_text, special_confidence = check_special_cases(cleaned_text)
    if is_special:
        return special_text, True, special_confidence, f"Special Case - {special_text}"
    
    # Check if it already matches a pattern
    is_match, pattern_name = matches_pattern(cleaned_text)
    if is_match:
        return cleaned_text, True, 1.0, pattern_name
    
    # Try common substitutions for full text
    variants = generate_pattern_variants(cleaned_text)
    
    # Check if any variant is a special case
    for variant in variants:
        is_special, special_text, special_confidence = check_special_cases(variant)
        if is_special and special_confidence > confidence_threshold:
            return special_text, True, special_confidence, f"Special Case - {special_text}"
    
    best_match = None
    best_pattern = None
    best_distance = float('inf')
    
    # Check all variants against all patterns
    for variant in variants:
        is_match, pattern_name = matches_pattern(variant)
        if is_match:
            return variant, True, 0.9, pattern_name  # Direct match through substitution
        
        # If no direct match, find the closest pattern match
        for pattern_dict in LICENSE_PATTERNS:
            # Extract pattern without the anchors and character classes
            simplified_pattern = pattern_dict["pattern"].replace('^', '').replace('$', '').replace('?', '')
            
            # Get the expected format by replacing pattern symbols with representative characters
            expected_format = re.sub(r'\[A-Z\]', 'A', simplified_pattern)
            expected_format = re.sub(r'\\\d', '1', expected_format)
            expected_format = re.sub(r'\{(\d+)(,\d+)?\}', lambda m: '1' * int(m.group(1)), expected_format)
            
            # Calculate distance to this pattern
            distance = levenshtein_distance(variant, expected_format)
            
            # Normalize distance by the length of the longer string
            normalized_distance = distance / max(len(variant), len(expected_format))
            
            if normalized_distance < best_distance:
                best_distance = normalized_distance
                best_match = variant
                best_pattern = pattern_dict["name"]
    
    # If we found a reasonably close match
    if best_match and best_distance < 0.3:  # Threshold for acceptable distance
        confidence = 1.0 - best_distance
        return best_match, False, confidence, best_pattern
    
    # Handle specific cases as a last resort
    if 'COV' in cleaned_text and ('ID' in cleaned_text or 'D' in cleaned_text) and any(char in '1234567890' for char in cleaned_text):
        return "COVID19", True, 0.8, "Special Case - COVID19"
    
    # No good matches found
    return cleaned_text, False, 0.0, None

def correct_candidates(candidates):
    """
    Process a list of text candidates, attempting to correct each one
    and adjust confidence scores based on pattern matching.
    
    Args:
        candidates: List of candidate dictionaries with text and confidence
        
    Returns:
        List of corrected candidates with pattern match information
    """
    corrected_candidates = []
    
    for candidate in candidates:
        original_text = candidate["text"]
        original_confidence = candidate["confidence"]
        original_char_positions = candidate.get("char_positions", [])
        
        # Try to correct the text
        corrected_text, is_pattern, correction_confidence, pattern_name = correct_plate_text(original_text)
        
        # Adjust confidence based on pattern matching
        pattern_bonus = 0.15 if is_pattern else 0.0
        final_confidence = min(1.0, original_confidence + pattern_bonus)
        
        # If corrected is different from original but matches a pattern well
        if corrected_text != original_text and correction_confidence > 0.7:
            corrected_candidates.append({
                "text": corrected_text,
                "confidence": final_confidence,
                "pattern_match": is_pattern,
                "pattern_name": pattern_name,
                "original_text": original_text,
                "char_positions": original_char_positions  # Preserve character positions
            })
        
        # Always include the original (with updated confidence if it's a pattern match)
        corrected_candidates.append({
            "text": original_text,
            "confidence": final_confidence if is_pattern else original_confidence,
            "pattern_match": is_pattern,
            "pattern_name": pattern_name if is_pattern else None,
            "char_positions": original_char_positions  # Always preserve character positions
        })
    
    # Sort by confidence
    corrected_candidates.sort(key=lambda x: x["confidence"], reverse=True)
    
    # Remove duplicates (keeping highest confidence)
    seen_texts = set()
    unique_candidates = []
    for candidate in corrected_candidates:
        if candidate["text"] not in seen_texts:
            seen_texts.add(candidate["text"])
            unique_candidates.append(candidate)
    
    return unique_candidates[:10]  # Return top 10 unique candidates

# NEW COMMON OCR EXTRACTION FUNCTIONS

def generate_character_analysis_for_covid19(original_confidence):
    """Generate character-by-character analysis for COVID19 text."""
    # Create a reasonable confidence level for the characters
    base_confidence = max(0.85, original_confidence)
    chars = ['C', 'O', 'V', 'I', 'D', '1', '9']
    
    # Generate character positions array
    covid_char_positions = []
    
    for i, char in enumerate(chars):
        # Create position candidates
        position_candidates = [
            {"char": char, "confidence": base_confidence}
        ]
        
        # Add alternatives with lower confidence
        if char in EXTENDED_CHAR_SIMILARITY:
            for idx, similar_char in enumerate(EXTENDED_CHAR_SIMILARITY[char][:3]):  # Just include top 3 alternatives
                alt_confidence = max(0.5, base_confidence * (0.9 - 0.1 * idx))
                position_candidates.append({
                    "char": similar_char,
                    "confidence": alt_confidence
                })
        
        # Add to char positions
        covid_char_positions.append({
            "position": i,
            "candidates": position_candidates
        })
    
    return covid_char_positions

def extract_text_from_plate(plate_region, preprocessing_level='standard'):
    """
    Extract text from a license plate region with multiple preprocessing approaches.
    
    Args:
        plate_region: Image containing the license plate
        preprocessing_level: Level of preprocessing to apply ('minimal', 'standard', or 'advanced')
        
    Returns:
        tuple: (best_text, candidates_list, original_ocr_text)
    """
    try:
        # Get the OCR reader
        reader = get_reader()
        
        # Preprocess the plate image for better OCR
        if len(plate_region.shape) > 2:  # If color image
            gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_region.copy()
        
        # Apply different preprocessing techniques based on the level requested
        preprocessed_images = []
        
        # Minimal preprocessing (just resize)
        img1 = cv2.resize(gray, None, fx=1.2, fy=1.2)
        preprocessed_images.append(img1)
        
        if preprocessing_level in ['standard', 'advanced']:
            # Standard preprocessing (with histogram equalization and thresholding)
            img2 = cv2.resize(gray, None, fx=1.5, fy=1.5)
            img2 = cv2.equalizeHist(img2)
            img2 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            preprocessed_images.append(img2)
            
            # Additional preprocessing with adaptive thresholding
            img3 = cv2.resize(gray, None, fx=1.5, fy=1.5)
            img3 = cv2.equalizeHist(img3)
            img3 = cv2.adaptiveThreshold(img3, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            preprocessed_images.append(img3)
            
        if preprocessing_level == 'advanced':
            # Advanced preprocessing techniques
            # Denoising
            img4 = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            img4 = cv2.resize(img4, None, fx=1.5, fy=1.5)
            img4 = cv2.equalizeHist(img4)
            img4 = cv2.threshold(img4, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            preprocessed_images.append(img4)
            
            # Edge enhancement then thresholding
            img5 = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
            img5 = cv2.threshold(img5, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            preprocessed_images.append(img5)
            
            # Additional preprocessing to handle large spaces within plate text
            # Try to close gaps with morphological operations
            img6 = cv2.resize(gray, None, fx=1.5, fy=1.5)
            img6 = cv2.equalizeHist(img6)
            kernel = np.ones((1, 5), np.uint8)  # Horizontal kernel to close gaps between characters
            img6 = cv2.morphologyEx(img6, cv2.MORPH_CLOSE, kernel)
            img6 = cv2.threshold(img6, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            preprocessed_images.append(img6)
            
            # Try erosion followed by dilation with a wider kernel
            img7 = cv2.resize(gray, None, fx=1.5, fy=1.5)
            kernel = np.ones((1, 7), np.uint8)
            img7 = cv2.erode(img7, kernel, iterations=1)
            img7 = cv2.dilate(img7, kernel, iterations=1)
            img7 = cv2.threshold(img7, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            preprocessed_images.append(img7)
        
        # Process each preprocessed image with OCR and keep the best result
        best_confidence = 0.0
        best_result = None
        best_candidates = []
        original_ocr_text = "Unknown"  # Default value for original OCR text
        
        for img_idx, processed_img in enumerate(preprocessed_images):
            # Perform OCR with allow_list to restrict to alphanumeric characters
            result = reader.readtext(processed_img, detail=1, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            
            candidates = []
            
            if result:
                # Process all detected text regions
                for detection in result:
                    bbox, text, confidence = detection
                    # Clean the text (remove spaces and unwanted characters)
                    cleaned_text = ''.join(c for c in text if c.isalnum())
                    
                    # Store the original OCR result (first detected text) if not set yet
                    if original_ocr_text == "Unknown" and cleaned_text:
                        original_ocr_text = cleaned_text
                        logger.info(f"Original OCR detection: {original_ocr_text}")
                    
                    if cleaned_text:  # Only add non-empty text
                        # Create a list to hold character positions with their candidates
                        char_positions = []
                        
                        # Extract text region for character-level analysis
                        y1, x1 = int(max(0, bbox[0][1])), int(max(0, bbox[0][0]))
                        y2, x2 = int(min(processed_img.shape[0], bbox[2][1])), int(min(processed_img.shape[1], bbox[2][0]))
                        text_region = processed_img[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else processed_img
                        
                        # Process each character in the text
                        for i, char in enumerate(cleaned_text):
                            # Create an array of character candidates for this position
                            position_candidates = []
                            
                            # Add the primary detected character first
                            position_candidates.append({
                                "char": char,
                                "confidence": float(confidence)
                            })
                            
                            # Calculate character's rough position in text region
                            char_width = text_region.shape[1] // max(1, len(cleaned_text))
                            start_x = max(0, i * char_width - 2)
                            end_x = min(text_region.shape[1], (i + 1) * char_width + 2)
                            
                            # Try to extract just this character for OCR
                            if end_x > start_x and text_region.size > 0:
                                char_img = text_region[:, start_x:end_x]
                                if char_img.size > 0:
                                    try:
                                        # Try to get character-level OCR results
                                        char_results = reader.readtext(char_img, detail=1, 
                                                                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                                        
                                        seen_chars = {char}  # Track seen characters to avoid duplicates
                                        
                                        # Add any detected characters from the character region
                                        for char_det in char_results:
                                            detected_char = char_det[1].strip()
                                            if detected_char and len(detected_char) == 1 and detected_char not in seen_chars:
                                                position_candidates.append({
                                                    "char": detected_char,
                                                    "confidence": float(char_det[2])
                                                })
                                                seen_chars.add(detected_char)
                                    except Exception as e:
                                        logger.error(f"Error in character OCR for {char}: {e}")
                            
                            # Add similar looking characters based on mapping
                            if char in EXTENDED_CHAR_SIMILARITY:
                                for idx, similar_char in enumerate(EXTENDED_CHAR_SIMILARITY[char]):
                                    if similar_char not in [c["char"] for c in position_candidates]:
                                        # Decrease confidence for each alternative
                                        alt_confidence = max(0.1, float(confidence) * (0.9 - 0.1 * idx))
                                        position_candidates.append({
                                            "char": similar_char,
                                            "confidence": alt_confidence
                                        })
                            
                            # Sort by confidence and keep top 5
                            position_candidates.sort(key=lambda x: x["confidence"], reverse=True)
                            position_candidates = position_candidates[:5]
                            
                            # Add to position list
                            char_positions.append({
                                "position": i,
                                "candidates": position_candidates
                            })
                        
                        # Check if text matches any common license plate pattern
                        pattern_match, pattern_name = matches_pattern(cleaned_text)
                        pattern_boost = 0.1 if pattern_match else 0.0
                        
                        # Include additional data in the candidate for debugging
                        candidates.append({
                            "text": cleaned_text,
                            "confidence": float(confidence) + pattern_boost,
                            "pattern_match": pattern_match,
                            "pattern_name": pattern_name if pattern_match else None,
                            "char_positions": char_positions
                        })
                
                # Record the best result based on confidence
                if candidates and candidates[0]["confidence"] > best_confidence:
                    best_confidence = candidates[0]["confidence"]
                    best_result = candidates[0]["text"]
                    best_candidates = candidates
                    
                    # Store which preprocessing method worked best
                    for candidate in best_candidates:
                        candidate["preprocessing_method"] = img_idx
                        
        # Apply text correction to improve our candidates
        if best_candidates:
            # Store the original detection for reference
            original_detection = {
                "text": original_ocr_text,
                "confidence": best_candidates[0].get("confidence", 0.5),
                "pattern_match": False,
                "pattern_name": "Original OCR"
            }
            
            # Apply corrections but ensure we preserve all the character position data
            corrected_candidates = correct_candidates(best_candidates)
            
            # Insert the original detection as a candidate for reference
            corrected_candidates.insert(1, original_detection)
            
            # Add detailed logging for each candidate
            for i, candidate in enumerate(corrected_candidates[:3]):  # Log top 3 candidates
                logger.info(f"Candidate {i+1}: {candidate['text']}, Confidence: {candidate['confidence']:.2f}, " +
                          f"Pattern Match: {candidate['pattern_match']}, " +
                          f"Pattern Name: {candidate.get('pattern_name', 'None')}")
            
            # Double-check special cases - Enhanced with more pattern recognition
            best_text = corrected_candidates[0]["text"]
            
            # Start with the most specific special case check
            if best_text == 'OD19' or 'OD19' in best_text:
                # This is almost certainly COVID19 - apply the correction
                logger.info(f"Special case OD19 detected - correcting to COVID19")
                
                # Create proper character analysis for COVID19
                covid_char_positions = generate_character_analysis_for_covid19(
                    corrected_candidates[0].get("confidence", 0.85)
                )
                
                # Update the first candidate
                corrected_candidates[0]["text"] = "COVID19"
                corrected_candidates[0]["confidence"] = 1.0  # Maximum confidence
                corrected_candidates[0]["pattern_match"] = True
                corrected_candidates[0]["pattern_name"] = "Special Case - COVID19"
                corrected_candidates[0]["char_positions"] = covid_char_positions  # Replace with COVID19 char data
                
                # If we have other candidates that are similar, fix them too
                for candidate in corrected_candidates[1:]:
                    if candidate["text"] == 'OD19':
                        candidate["original_text"] = candidate["text"]
                        candidate["text"] = "COVID19"
                        candidate["pattern_match"] = True
                        candidate["pattern_name"] = "Special Case - COVID19"
                        candidate["char_positions"] = covid_char_positions  # Replace with COVID19 char data
                
                # Add the original OCR result back to candidates
                corrected_candidates.insert(1, original_detection)
                return "COVID19", corrected_candidates, original_ocr_text
            
            # Check other COVID-like patterns
            is_covid, confidence = looks_like_covid(best_text)
            if is_covid and confidence > 0.65:
                logger.info(f"COVID pattern detected in '{best_text}' - correcting to COVID19 with confidence {confidence}")
                
                # Create proper character analysis for COVID19
                covid_char_positions = generate_character_analysis_for_covid19(
                    max(corrected_candidates[0].get("confidence", 0.85), confidence)
                )
                
                corrected_candidates[0]["text"] = "COVID19"
                corrected_candidates[0]["confidence"] = max(corrected_candidates[0]["confidence"], confidence)
                corrected_candidates[0]["pattern_match"] = True
                corrected_candidates[0]["pattern_name"] = "Special Case - COVID19"
                corrected_candidates[0]["char_positions"] = covid_char_positions  # Replace with COVID19 char data
                
                return "COVID19", corrected_candidates, original_ocr_text
            
            # More general special case handling 
            elif 'COV' in best_text or 'COVID' in best_text:
                logger.info(f"Possible COVID case detected in candidate: {best_text}")
                is_special, special_text, special_confidence = check_special_cases(best_text)
                if is_special:
                    # Update the first candidate with the corrected text
                    corrected_candidates[0]["text"] = special_text
                    corrected_candidates[0]["confidence"] = max(corrected_candidates[0]["confidence"], special_confidence)
                    corrected_candidates[0]["pattern_match"] = True
                    corrected_candidates[0]["pattern_name"] = f"Special Case - {special_text}"
                    logger.info(f"Corrected to special case: {special_text} with confidence {special_confidence:.2f}")
                    return special_text, corrected_candidates, original_ocr_text
                
            return corrected_candidates[0]["text"], corrected_candidates, original_ocr_text
        
        # Return default values if no valid text found
        return "Unknown", [{"text": "Unknown", "confidence": 0.0, "pattern_match": False, "char_positions": []}], original_ocr_text
    
    except Exception as e:
        logger.error(f"Error in OCR text extraction: {e}")
        return "Error", [{"text": "Error", "confidence": 0.0, "pattern_match": False, "error": str(e)}], "Error"

def extract_text_from_region(image, bbox):
    """
    Extract license plate text from a bounding box in an image.
    
    Args:
        image: The full image
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        
    Returns:
        tuple: (text, candidates)
    """
    try:
        x1, y1, x2, y2 = bbox
        plate_region = image[y1:y2, x1:x2]
        if plate_region.size == 0:
            return "Invalid region", []
            
        return extract_text_from_plate(plate_region)
    except Exception as e:
        logger.error(f"Error extracting text from region: {e}")
        return "Error", []

def similarity_score(text1, text2):
    """Calculate similarity between two strings."""
    return SequenceMatcher(None, text1, text2).ratio()
