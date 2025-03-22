import os
import cv2
import numpy as np
import tensorflow as tf
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VehicleOrientationDetector:
    def __init__(self):
        # Fix the path calculation
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(current_dir, 'VOI', 'models')
        # Update to use .keras file instead of .h5
        self.model_path = os.path.join(models_dir, 'orientation_model.keras')
        
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the orientation detection model."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found at: {self.model_path}")
            
            logger.info(f"Loading model from: {self.model_path}")
            self.model = tf.keras.models.load_model(
                self.model_path,
                compile=False,
            )
            logger.info("Successfully loaded model")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def get_model_input_shape(self):
        """Get the expected input shape from the model."""
        try:
            input_shape = self.model.layers[0].input_shape
            if isinstance(input_shape, list):
                input_shape = input_shape[0]
            
            if input_shape and len(input_shape) == 4:
                return input_shape[1:3]
        except:
            logger.warning("Could not determine model input shape, using default (224, 224)")  # Updated default size
        return (224, 224)  # Updated default size to match model architecture

    def preprocess_image(self, image):
        """
        Preprocess an image for orientation prediction.
        """
        if image is None:
            logger.error("Invalid input image")
            return None
        
        # Convert from BGR to RGB (Keras models typically use RGB)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image to target size
        target_size = self.get_model_input_shape()
        img = cv2.resize(img, target_size)
        
        # Normalize pixel values to [0, 1]
        img = img / 255.0
        return img

    def predict(self, image):
        """
        Predict if a vehicle is facing toward the camera (front) or away (rear).
        Returns a dict with orientation and confidence.
        """
        try:
            if self.model is None:
                raise ValueError("Model not loaded")

            # Preprocess the image
            img = self.preprocess_image(image)
            if img is None:
                raise ValueError("Failed to preprocess image")

            # Add batch dimension
            img_batch = np.expand_dims(img, axis=0)

            # Make prediction
            prediction = self.model.predict(img_batch, verbose=0)[0][0]

            # Use more precise thresholding from test script
            threshold = 0.15  # Lower threshold since raw values tend to be small
            is_front = prediction >= threshold
            
            # Calculate confidence based on prediction value
            if is_front:
                confidence = min((prediction / threshold) * 100, 100) / 100.0
            else:
                confidence = min(((threshold - prediction) / threshold) * 100, 100) / 100.0
            
            # Convert to final output format
            orientation = "Front-facing" if is_front else "Rear-facing"
            logger.info(f"Raw prediction value: {prediction:.4f}")
            logger.info(f"Threshold: {threshold}")
            logger.info(f"Is front: {is_front}")
            logger.info(f"Calculated confidence: {confidence:.2%}")
            
            return {
                "orientation": orientation,
                "confidence": float(confidence),
                "is_front": bool(is_front)
            }

        except Exception as e:
            logger.error(f"Error predicting vehicle orientation: {e}")
            return {
                "orientation": "Unknown",
                "confidence": 0.0,
                "is_front": None,
                "error": str(e)
            }

# Create a singleton instance
vehicle_orientation_detector = VehicleOrientationDetector()
