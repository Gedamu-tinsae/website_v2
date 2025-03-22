# Car Plate Extractor Project Structure

## Frontend
`frontend/`
├── public/
│   ├── index.html
|   |── fav.ico
│   └── ...
├── src/
│   ├── assets/
│   │   ├── video-icon.png
│   │   ├── file-icon.png
│   │   ├── realtime-icon.png
│   │   ├── processing-icon.png
│   │   ├── reload-icon.png
│   │   └── expand-icon.png ...
│   ├── components/
│   │   ├── Navbar.js
│   │   └── ...
│   ├── pages/
│   │   ├── AboutPage.js
│   │   ├── DbPage.js
│   │   ├── DocsPage.js
│   │   ├── HomePage.js
│   │   └── RealtimeDetection.js
│   ├── styles/
│   │   ├── App.css
│   │   ├── HomePage.css
│   │   ├── Navbar.css
│   │   └── index.css
│   ├── App.js
│   ├── index.js
│   └── ...
├── .gitignore
└── ...

## Backend
`backend/`
├── app/
│   ├── api/
│   │   ├── endpoints/
│   │   │   ├── upload.py
│   │   │   ├── upload_video.py
│   │   │   ├── upload_tensorflow.py 
│   │   │   ├── realtime.py
│   │   │   ├── example.json
│   │   │   └── ...
│   ├── models/
│   │   ├── opencv_tensorflow.py
│   │   ├── tensorflow_model.py  
│   │   ├── plate_correction.py
│   │   ├── vehicle_orientation.py
│   │   ├── haze_removal.py     # Haze/Fog Removal implementation
│   │   ├── vehicle_type.py     # Vehicle Type Detection using YOLOv8
│   │   ├── vehicle_make.py
│   │   ├── color_detection.py
│   │   ├── gf.py               
|   |   ├── saved_model/        # plate detection model
|   |   |   └── saved_model.pb
│   │   ├── VTD/                # Vehicle and Traffic Detection
│   │   │   └── yolov8n.pt      
│   │   ├── VOI/                # Vehicle Orientation Identification
│   │   │   ├── models/
│   │   │   │   ├── orientation_model.h5
│   │   │   └── └── orientation_model.keras
│   │   └── VMI/                # Vehicle Make Identification
│   │       ├── models/         
│   │       │   ├── make_classes.json         
│   │       │   ├── make_mobilenet_weights.h5 
│   │       └── └── final_make_mobilenet_weights.h5 
│   └── main.py
├── results/
│   └── ... (generated result images and videos)
├── uploads/
│   └── ... (uploaded images)
├── venv/
│   └── ...
└── ...

## Root
- `file_structure.md`
- Readme.md
