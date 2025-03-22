# Car Plate Extractor & Vehicle Analysis System

A comprehensive full-stack application for extracting license plates and analyzing vehicle characteristics from images and videos. The system combines computer vision, deep learning, and advanced image processing techniques to provide detailed vehicle analysis beyond just license plate recognition.


## Features

- **License Plate Recognition**: Extract and interpret text from vehicle license plates in images and videos
- **Vehicle Analysis**: 
  - Detect vehicle type (car, truck, bus, motorcycle, etc.)
  - Identify vehicle color with color distribution analysis
  - Determine vehicle orientation (front, rear, side)
  - Recognize vehicle make (manufacturer)
- **Multiple Processing Methods**: Choose between OpenCV and TensorFlow-based processing pipelines
- **Real-time Detection**: Process live video streams for immediate analysis
- **Enhanced OCR**: Pattern matching and correction algorithms for improved accuracy
- **Visualization Tools**: View intermediate processing steps and detailed results
- **Fog/Haze Removal**: Special processing for low visibility conditions
- **Comprehensive Analytics**: Color distribution charts, confidence scores, and alternative detections

### Low Visibility & Adverse Weather Processing

- **Haze/Fog Detection**: Automatically identifies images captured in poor visibility conditions
- **Image Dehazing**: Implements the Dark Channel Prior algorithm to recover clear images from hazy/foggy scenes
- **Guided Filtering**: Preserves edge details while enhancing clarity in low-contrast regions
- **Weather-Adaptive Processing**: Different processing pipelines optimized for various weather conditions
- **Contrast Enhancement**: Specialized algorithms to improve visibility in overexposed or underexposed images
- **Interactive Controls**: Users can manually enable dehazing for specific images or videos

## Advanced Vehicle Analysis

### Vehicle Type Detection
- Identifies common vehicle types including cars, trucks, buses, motorcycles, and more
- Provides confidence scores and alternative possibilities
- Uses both full-image and region-based detection for improved accuracy

### Vehicle Color Analysis
- Detects primary vehicle color with confidence scores
- Provides color distribution percentages
- Includes color visualization with RGB/hex values
- Differentiates between full-image and vehicle-region color detection

### Vehicle Orientation
- Determines if the vehicle is captured from the front, rear, or side
- Helps in contextualizing license plate location and visibility
- Improves accuracy of other detection systems

### Vehicle Make Identification
- Recognizes the manufacturer/brand of vehicles
- Provides confidence scores and alternative possibilities
- Uses specialized deep learning models trained on vehicle makes

## Architecture

The application consists of two main components:

1. **Frontend**: A React-based web interface for uploading images/videos and displaying results
2. **Backend**: A FastAPI server that handles image processing, computer vision tasks, and machine learning models

## Getting Started

### Prerequisites

- Python 3.8+ (backend)
- Node.js 14+ (frontend)
- npm or yarn (frontend package management)

### Backend Setup

1. **Clone the Repository**
   ```bash
   git clone hhttps://github.com/Gedamu-tinsae/website_v2.git
   cd car-plate-extractor/backend
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   ```

   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Backend Server**
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```
   The server will start on `http://localhost:8000`

### Frontend Setup

1. **Navigate to the Frontend Directory**
   ```bash
   cd ../frontend
   ```

2. **Install Dependencies**
   ```bash
   npm install
   ```

3. **Start the Development Server**
   ```bash
   npm start
   ```
   The frontend will be available at `http://localhost:3000`

## Usage

### Image Upload and Processing

1. Navigate to the home page
2. Select your preferred upload method (file, video, real-time)
3. Upload an image or video containing a vehicle
4. View the extracted license plate text and comprehensive vehicle analysis
5. Explore the detailed visualizations and intermediate processing steps

### Video Processing

Video processing may take longer depending on the length and resolution of the video. The system processes videos at approximately 1 frame per 3 seconds for high-quality analysis. For a 10-second video at 30fps, processing may take around 10-15 minutes.

### Real-time Detection

The real-time detection feature allows you to use your webcam or connected camera to detect license plates and analyze vehicles on the fly. This mode uses optimized algorithms for faster processing at a slight reduction in accuracy.

### Processing in Adverse Weather Conditions

The system includes specialized processing for images and videos captured in foggy, hazy, or low-visibility conditions:

1. Toggle the "Enable Low Visibility Processing" option when uploading
2. The system will apply dehazing algorithms before performing license plate detection
3. View the intermediate dehazing steps in the detailed results
4. Compare original and dehazed images to see visibility improvements
5. Processing time may increase slightly when dehazing is enabled

## Project Structure

### Frontend

```
frontend/
├── public/               # Static files
├── src/
│   ├── assets/           # Icons and images
│   ├── components/       # Reusable components
│   ├── pages/            # Page components
│   ├── styles/           # CSS files
│   ├── App.js            # Main application component
│   └── index.js          # Entry point
└── package.json          # Dependencies and scripts
```

### Backend

```
backend/
├── app/
│   ├── api/              # API endpoints
│   │   └── endpoints/    # Individual API routes
│   ├── models/           # Processing models and algorithms
│   └── main.py           # FastAPI application
├── results/              # Generated result images and videos
├── uploads/              # Temporary storage for uploaded files
└── requirements.txt      # Python dependencies
```

## Technologies Used

### Frontend
- React.js
- React Router
- CSS3

### Backend
- FastAPI
- OpenCV
- TensorFlow
- PyTorch (YOLOv8)
- EasyOCR
- NumPy
- Scikit-learn (for color clustering)

## Model Details

The system utilizes several specialized models and algorithms:

1. **License Plate Detection**: 
   - TensorFlow object detection model trained on license plate datasets
   - Specialized preprocessing for different lighting conditions

2. **Vehicle Type Detection**: 
   - YOLOv8 model trained on diverse vehicle types
   - Categories include sedan, SUV, truck, bus, motorcycle, etc.
   - Located in the `VTD` (Vehicle and Traffic Detection) module

3. **Vehicle Orientation Identification**: 
   - Custom CNN model to determine vehicle facing direction
   - Identifies front, rear, and side views
   - Located in the `VOI` module with Keras/TensorFlow implementation

4. **Vehicle Make Identification**: 
   - MobileNet-based classification model
   - Trained to identify major vehicle manufacturers
   - Located in the `VMI` module with JSON class mapping

5. **Color Detection**: 
   - K-means clustering for dominant color extraction
   - Color naming algorithm for human-readable color labels
   - Confidence scoring based on color distribution

6. **Haze/Fog Removal**:
   - Dark channel prior algorithm for dehazing
   - Guided filtering for edge preservation
   - Enhances visibility in adverse weather conditions
   - Multi-scale processing for different fog densities
   - Transmission map estimation for realistic atmospheric light calculation
   - Preserves color fidelity while removing haze effects

## Performance Considerations

- Processing a single image takes approximately 3-5 seconds
- Video processing is resource-intensive, with a 10-second video (30fps) taking approximately 10-15 minutes to process fully
- Real-time detection operates at reduced accuracy for better performance
- The system uses region-based optimization to focus computational resources on areas of interest

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Special thanks to the contributors and open-source libraries that made this project possible
- YOLOv8 developers for the vehicle detection models
- EasyOCR team for the optical character recognition capabilities