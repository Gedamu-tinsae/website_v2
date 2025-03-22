from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from app.models.tensorflow_model import process_image_with_model, process_video_with_model
from app.models.haze_removal import HazeRemoval
import shutil
import os
import logging
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/upload_image_tensorflow")
async def upload_image_tensorflow(file: UploadFile = File(...), low_visibility: bool = Form(False)):
    try:
        # Save the uploaded file
        upload_dir = os.path.join("uploads", "tensorflow", "images")
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Image uploaded successfully: {file_path}")

        # Apply dehazing if low_visibility is True
        if low_visibility:
            logger.info("Applying dehazing to low visibility image")
            # Create a temporary path for dehazed image
            dehazed_path = os.path.join(upload_dir, f"dehazed_{file.filename}")
            
            # Apply dehazing using HazeRemoval class
            hr = HazeRemoval()
            hr.open_image(file_path)
            hr.get_dark_channel()
            hr.get_air_light()
            hr.get_transmission()
            hr.guided_filter()
            hr.recover()
            
            # Get all intermediate images
            dehaze_stages = hr.get_all_intermediate_images()
            
            # Save dehazed image
            cv2.imwrite(dehazed_path, hr.dst)
            logger.info(f"Dehazed image saved at: {dehazed_path}")
            
            # Process the dehazed image with the TensorFlow model
            result = process_image_with_model(dehazed_path, confidence_threshold=0.6)
            
            # Add dehazing info and intermediate images to result
            result["preprocessing"] = "dehazing_applied"
            result["original_path"] = file_path
            
            # Convert all intermediate dehazing images to base64
            import base64
            dehaze_intermediate_base64 = {}
            for key, img in dehaze_stages.items():
                _, buffer = cv2.imencode('.jpg', img)
                dehaze_intermediate_base64[key] = base64.b64encode(buffer).decode('utf-8')
            
            # Add the dehazing intermediate images to the result
            result["dehaze_stages"] = dehaze_intermediate_base64
            
        else:
            # Process the image with the TensorFlow model without dehazing
            result = process_image_with_model(file_path, confidence_threshold=0.6 if low_visibility else 0.7)

        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error uploading image: {e}")
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

@router.post("/upload_video_tensorflow")
async def upload_video_tensorflow(file: UploadFile = File(...), low_visibility: bool = Form(False)):
    # For videos, we'll handle low visibility processing frame by frame in the model
    try:
        # Save the uploaded file
        upload_dir = os.path.join("uploads", "tensorflow", "videos")
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Video uploaded successfully: {file_path}")

        # Process the video with the TensorFlow model
        # Pass a lower confidence threshold for low visibility
        confidence_threshold = 0.6 if low_visibility else 0.7
        result = process_video_with_model(file_path, low_visibility=low_visibility, confidence_threshold=confidence_threshold)

        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error uploading video: {e}")
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)
