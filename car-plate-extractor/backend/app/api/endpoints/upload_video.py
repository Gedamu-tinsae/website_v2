from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from app.models.opencv_tensorflow import process_video
import shutil
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    try:
        # Save the uploaded file
        upload_dir = os.path.join("uploads", "opencv", "videos")
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Video uploaded successfully: {file_path}")

        # Process the video
        result = process_video(file_path)

        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error uploading video: {e}")
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)
