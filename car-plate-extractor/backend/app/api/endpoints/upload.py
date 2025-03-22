from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from app.models.opencv_tensorflow import process_image
from app.models.haze_removal import HazeRemoval
import shutil
import os
import logging
import cv2
import numpy as np
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...), low_visibility: bool = Form(False)):
    try:
        # Save the uploaded file
        upload_dir = os.path.join("uploads", "opencv", "images")
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"File uploaded successfully: {file_path}")

        # Apply dehazing if low_visibility is True
        if low_visibility:
            logger.info("Applying dehazing to low visibility image")
            # Create a temporary path for dehazed image
            dehazed_path = os.path.join(upload_dir, f"dehazed_{file.filename}")
            
            # Load original image for later use
            original_image = cv2.imread(file_path)
            
            # Apply dehazing using HazeRemoval class
            hr = HazeRemoval()
            hr.open_image(file_path)
            hr.get_dark_channel()
            hr.get_air_light()
            hr.get_transmission()
            hr.guided_filter()
            hr.recover()
            
            # Get all intermediate images (except the original which is redundant)
            dehaze_stages = hr.get_all_intermediate_images()
            # Remove the original from stages as it's redundant
            if 'original' in dehaze_stages:
                del dehaze_stages['original']
            
            # Save dehazed image
            cv2.imwrite(dehazed_path, hr.dst)
            logger.info(f"Dehazed image saved at: {dehazed_path}")
            
            # Process the dehazed image with a lower confidence threshold
            result = process_image(dehazed_path, confidence_threshold=0.6)
            
            # Add dehazing info to result
            result["preprocessing"] = "dehazing_applied"
            result["original_path"] = file_path
            
            # Convert all intermediate dehazing images to base64
            dehaze_intermediate_base64 = {}
            for key, img in dehaze_stages.items():
                # Ensure all images are in BGR before encoding (OpenCV default)
                if img.shape[2] == 3:  # If it's a 3-channel image
                    # No need for conversion if already in BGR format
                    # But we should check the format to be certain
                    if key == "dehazed": 
                        logger.info(f"Dehazed image shape: {img.shape}, min: {img.min()}, max: {img.max()}")
                        # If it's still mismatched, try direct BGR to RGB conversion
                        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Encode the image
                _, buffer = cv2.imencode('.jpg', img)
                dehaze_intermediate_base64[key] = base64.b64encode(buffer).decode('utf-8')
            
            # Add the dehazing intermediate images to the result
            result["dehaze_stages"] = dehaze_intermediate_base64
            
            # Remove the code that tries to restore original colors to the annotated image
            # This will keep the dehazed image with annotations
            
        else:
            # Process the image without dehazing
            result = process_image(file_path, confidence_threshold=0.7)

        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)
