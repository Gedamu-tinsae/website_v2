import React, { useState, useEffect, useRef } from 'react';
import '../styles/HomePage.css';
import videoIcon from '../assets/video-icon.png';
import realtimeIcon from '../assets/realtime-icon.png';
import fileIcon from '../assets/file-icon.png';
import processingIcon from '../assets/processing-icon.png';
import reloadIcon from '../assets/reload-icon.png';
import expandIcon from '../assets/expand-icon.png';
import RealtimeDetection from './RealtimeDetection';

const HomePage = () => {
  const [resultMedia, setResultMedia] = useState(null);
  const [originalMedia, setOriginalMedia] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isUploaded, setIsUploaded] = useState(false);
  const [selectedOption, setSelectedOption] = useState('');
  const [showMoreOriginal, setShowMoreOriginal] = useState(false);
  const [showMoreAnnotated, setShowMoreAnnotated] = useState(false);
  const [processingInfo, setProcessingInfo] = useState(null);
  const [intermediateImages, setIntermediateImages] = useState({});
  const [isExpandedOriginal, setIsExpandedOriginal] = useState(false);
  const [isExpandedAnnotated, setIsExpandedAnnotated] = useState(false);
  const [mediaType, setMediaType] = useState(''); // New state to track media type
  const [timer, setTimer] = useState(0);
  const [milliseconds, setMilliseconds] = useState(0);
  const [finalTime, setFinalTime] = useState(null);
  const [processingMethod, setProcessingMethod] = useState('opencv'); // New state for processing method
  const [isLowVisibility, setIsLowVisibility] = useState(false); // New state for low visibility toggle
  const startTimeRef = useRef(null);
  const [isRealtimeActive, setIsRealtimeActive] = useState(false);

  // Helper function to determine the confidence class for a character
  const getConfidenceClass = (confidence) => {
    if (confidence >= 0.7) return 'high-confidence';
    if (confidence >= 0.4) return 'medium-confidence';
    return 'low-confidence';
  };
  
  // Helper function to get bar colors for color visualization
  const getColorForBar = (colorName) => {
    // Map color names to CSS color values
    const colorMap = {
      'red': '#ff6b6b',
      'orange': '#ff9f43',
      'yellow': '#feca57',
      'green': '#1dd1a1',
      'blue': '#54a0ff',
      'purple': '#5f27cd',
      'white': '#c8d6e5',
      'black': '#222f3e',
      'gray': '#8395a7',
      'silver': '#dfe4ea',
      'brown': '#a0522d'
    };
    
    // Return the mapped color or a default if not found
    return colorMap[colorName] || '#6c5ce7';
  };

  useEffect(() => {
    let interval;
    if (isProcessing) {
      startTimeRef.current = Date.now();
      interval = setInterval(() => {
        const elapsedTime = Date.now() - startTimeRef.current;
        setTimer(Math.floor(elapsedTime / 1000));
        setMilliseconds(elapsedTime % 1000);
      }, 10);
    } else {
      setTimer(0);
      setMilliseconds(0);
    }
    return () => clearInterval(interval);
  }, [isProcessing]);

  useEffect(() => {
    if (!isProcessing && isUploaded) {
      const elapsedTime = Date.now() - startTimeRef.current;
      setFinalTime(`${Math.floor(elapsedTime / 1000)} seconds ${elapsedTime % 1000} milliseconds`);
    }
  }, [isProcessing, isUploaded]);

  const handleFileClick = (type) => {
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = type === 'video' ? 'video/*' : 'image/*';
    fileInput.onchange = async (event) => {
      const files = event.target.files;
      if (files.length > 0) {
        const formData = new FormData();
        formData.append('file', files[0]);
        // Add low visibility flag to the form data
        formData.append('low_visibility', isLowVisibility);
        const fileURL = URL.createObjectURL(files[0]);
        console.log('File URL:', fileURL);
        setOriginalMedia(fileURL);
        setMediaType(type);
        setIsProcessing(true);
        setIsUploaded(false); // Don't set to true until processing is complete

        try {
          const startTime = Date.now(); // Start timing
          const endpoint = type === 'video' 
            ? (processingMethod === 'tensorflow' ? 'upload_video_tensorflow' : 'upload_video') 
            : (processingMethod === 'tensorflow' ? 'upload_image_tensorflow' : 'upload');
          const response = await fetch(`http://172.20.10.10:8000/api/${endpoint}`, {
            method: 'POST',
            body: formData,
          });
          const result = await response.json();
          console.log('Upload result:', result);
          if (result.status === 'success') {
            // Calculate processing time
            const endTime = Date.now();
            const processingTime = endTime - startTime;
            setFinalTime(`${Math.floor(processingTime / 1000)} seconds ${processingTime % 1000} milliseconds`);
            
            setResultMedia(`http://172.20.10.10:8000${encodeURI(result.result_url)}`);
            setProcessingInfo(result);
            setIntermediateImages(result.intermediate_images);
            setIsUploaded(true); // Set to true only after successful processing
          }
        } catch (error) {
          console.error('Error uploading file:', error);
          setIsUploaded(false); // Ensure it's false if there's an error
        } finally {
          setIsProcessing(false);
        }
      }
    };
    fileInput.click();
  };

  const handleOptionChange = (event) => {
    const newMethod = event.target.value;
    setSelectedOption(newMethod);
    setProcessingMethod(newMethod);
    // If we're on the result page and a valid method is selected, automatically reload
    if (isUploaded && originalMedia && newMethod) {
      handleReloadClick(newMethod);
    }
  };

  const handleReloadClick = async (newMethod = processingMethod) => {
    if (originalMedia) {
      setIsProcessing(true);
      setIsUploaded(false);
      // Reset states before processing
      setResultMedia(null);
      setProcessingInfo(null);
      setIntermediateImages(null);
      setShowMoreOriginal(false);
      setShowMoreAnnotated(false);
      setFinalTime(null);
      
      try {
        const startTime = Date.now(); // Start timing
        const formData = new FormData();
        // Add low visibility flag to the form data
        formData.append('low_visibility', isLowVisibility);
        
        // Get the current media file
        let mediaBlob;
        if (originalMedia.startsWith('blob:')) {
          // If it's a blob URL from a file upload
          const response = await fetch(originalMedia);
          mediaBlob = await response.blob();
        } else {
          // If it's a file that was previously uploaded
          const response = await fetch(originalMedia);
          mediaBlob = await response.blob();
        }
        
        // Set the correct filename and type
        const filename = mediaType === 'video' ? 'video.mp4' : 'image.jpg';
        formData.append('file', mediaBlob, filename);
        
        // Use the new method parameter to determine the endpoint
        const endpoint = mediaType === 'video'
          ? (newMethod === 'tensorflow' ? 'upload_video_tensorflow' : 'upload_video')
          : (newMethod === 'tensorflow' ? 'upload_image_tensorflow' : 'upload');
          
        const uploadResponse = await fetch(`http://172.20.10.10:8000/api/${endpoint}`, {
          method: 'POST',
          body: formData,
        });
        const result = await uploadResponse.json();
        console.log('Reload result:', result);
        
        if (result.status === 'success') {
          // Calculate processing time
          const endTime = Date.now();
          const processingTime = endTime - startTime;
          setFinalTime(`${Math.floor(processingTime / 1000)} seconds ${processingTime % 1000} milliseconds`);
          
          // Update all states with new data
          setResultMedia(`http://172.20.10.10:8000${encodeURI(result.result_url)}`);
          setProcessingInfo(result);
          setIntermediateImages(result.intermediate_images);
          setIsUploaded(true);
        }
      } catch (error) {
        console.error('Error reloading file:', error);
        setIsUploaded(false);
      } finally {
        setIsProcessing(false);
      }
    }
  };

  const handleMoreClick = (type) => {
    if (type === 'original') {
      setShowMoreOriginal(!showMoreOriginal);
    } else if (type === 'annotated') {
      setShowMoreAnnotated(!showMoreAnnotated);
    }
  };

  const handleCloseClick = (type) => {
    if (type === 'original') {
      setShowMoreOriginal(false);
    } else if (type === 'annotated') {
      setShowMoreAnnotated(false);
    }
  };

  const handleExpandClick = (type) => {
    if (type === 'original') {
      setIsExpandedOriginal(!isExpandedOriginal);
      document.querySelector('.result-page').classList.toggle('blurred-border', !isExpandedOriginal);
    } else if (type === 'annotated') {
      setIsExpandedAnnotated(!isExpandedAnnotated);
      document.querySelector('.result-page').classList.toggle('blurred-border', !isExpandedAnnotated);
    }
  };

  const getMediaHeight = (mediaId) => {
    const media = document.getElementById(mediaId);
    return media ? media.clientHeight : 'auto';
  };

  const getMediaWidth = (mediaId) => {
    const media = document.getElementById(mediaId);
    return media ? media.clientWidth : 'auto';
  };

  const handlePlayClick = () => {
    const originalVideo = document.getElementById('originalMedia');
    const annotatedVideo = document.getElementById('annotatedMedia');
    console.log('Original Media URL:', originalMedia); // Debugging: Check the original media URL
    console.log('Annotated Media URL:', resultMedia); // Debugging: Check the annotated media URL
    if (originalVideo && annotatedVideo) {
      originalVideo.currentTime = 0;
      annotatedVideo.currentTime = 0;
      originalVideo.play().catch(error => console.error('Error playing original video:', error));
      annotatedVideo.play().catch(error => console.error('Error playing annotated video:', error));
    }
  };

  useEffect(() => {
    console.log('Original Media URL:', originalMedia); // Debugging: Check the original media URL
  }, [originalMedia]);

  const handleRealtimeClick = () => {
    setIsRealtimeActive(true);
  };

  const handleRealtimeClose = () => {
    setIsRealtimeActive(false);
  };

  if (isProcessing) {
    return (
      <div className="processing-page">
        <h2>Processing...</h2>
        <div className="processing-icon-container">
          <img src={processingIcon} alt="Processing Icon" className="processing-icon" />
        </div>
        <p>Time elapsed: {timer} seconds {milliseconds} milliseconds</p> {/* Add timer display */}
      </div>
    );
  }

  if (isUploaded && resultMedia) {
    return (
      <div className="result-page" style={{ height: showMoreOriginal || showMoreAnnotated ? 'auto' : '100vh' }}>
        <div className="four-img-container"> {/* New container */}
          <div className="result-images">
            <div className="image-container">
              <h3>Original {mediaType === 'video' ? 'Video' : 'Image'}:</h3>
              {originalMedia ? (
                mediaType === 'video' ? (
                  <video id="originalMedia" src={originalMedia} controls />
                ) : (
                  <img id="originalMedia" src={originalMedia} alt="Original" />
                )
              ) : (
                <div className="blank-canvas"></div> // Show blank canvas if original media is not available
              )}
              <button className="more-btn" onClick={() => handleMoreClick('original')}>
                More <span className="arrow">{showMoreOriginal ? '▲' : '▼'}</span>
              </button>
              {showMoreOriginal && (
                <div className={`more-window ${isExpandedOriginal ? 'expanded' : ''}`} style={{ backgroundColor: '#1b1924', height: isExpandedOriginal ? getMediaHeight('originalMedia') * 1.3 : getMediaHeight('originalMedia'), width: isExpandedOriginal ? getMediaWidth('originalMedia') * 1.3 : getMediaWidth('originalMedia') }}>
                  <button className="close-btn" onClick={() => handleCloseClick('original')}>×</button>
                  <button className="expand-btn" onClick={() => handleExpandClick('original')}>
                    <img src={expandIcon} alt="Expand Icon" style={{ width: '20px', height: '20px', filter: 'invert(100%) sepia(100%) saturate(0%) hue-rotate(60deg) brightness(100%) contrast(100%)' }} />
                  </button>
                  
                  {/* Add dehazing stages if present */}
                  {processingInfo && processingInfo.dehaze_stages && (
                    <div className="dehaze-stages">
                      <h3>Dehazing Process</h3>
                      
                      {/* Remove the original hazy image display since it's redundant */}
                      
                      <p><strong>Dark Channel:</strong></p>
                      <img src={`data:image/jpeg;base64,${processingInfo.dehaze_stages.dark_channel}`} alt="Dark Channel" />
                      
                      <p><strong>Transmission Map:</strong></p>
                      <img src={`data:image/jpeg;base64,${processingInfo.dehaze_stages.transmission}`} alt="Transmission Map" />
                      
                      <p><strong>Refined Transmission Map:</strong></p>
                      <img src={`data:image/jpeg;base64,${processingInfo.dehaze_stages.refined_transmission}`} alt="Refined Transmission" />
                      
                      <p><strong>Dehazed Result:</strong></p>
                      <img src={`data:image/jpeg;base64,${processingInfo.dehaze_stages.dehazed}`} alt="Dehazed Result" />
                    </div>
                  )}
                  
                  {/* Show OpenCV intermediate stages */}
                  {processingMethod === 'opencv' && intermediateImages && (
                    <div className="intermediate-images">
                      <h3>OpenCV Pipeline</h3>
                      <p><strong>Gray Scale:</strong></p>
                      <img src={`data:image/jpeg;base64,${intermediateImages.gray}`} alt="Gray Scale" />
                      <p><strong>Edge Detection:</strong></p>
                      <img src={`data:image/jpeg;base64,${intermediateImages.edge}`} alt="Edge Detection" />
                      <p><strong>Localized Image:</strong></p>
                      <img src={`data:image/jpeg;base64,${intermediateImages.localized}`} alt="Localized" />
                      <p><strong>License Plate:</strong></p>
                      <img src={`data:image/jpeg;base64,${intermediateImages.plate}`} alt="License Plate" />
                      
                      {/* Add Vehicle Region for color detection - improved with fallback text */}
                      <p><strong>Vehicle Region for Color Detection:</strong></p>
                      {intermediateImages.vehicle_region ? (
                        <img 
                          src={`data:image/jpeg;base64,${intermediateImages.vehicle_region}`} 
                          alt="Vehicle Region"
                          onError={(e) => {
                            console.log('Error loading vehicle region image');
                            e.target.style.display = 'none';
                            e.target.parentNode.innerHTML += '<div class="error-message">Vehicle region image could not be loaded. The region might be invalid or too small.</div>';
                          }}
                        />
                      ) : (
                        <div className="missing-region">
                          Vehicle region data not available. Color was detected using the full image.
                          {processingInfo && processingInfo.debug_info && (
                            <div className="debug-info">
                              <p>Debug info:</p>
                              <ul>
                                <li>Vehicle region exists: {processingInfo.debug_info.vehicle_region_exists ? 'Yes' : 'No'}</li>
                                <li>Region shape: {processingInfo.debug_info.vehicle_region_shape ? 
                                                   `${processingInfo.debug_info.vehicle_region_shape[0]}x${processingInfo.debug_info.vehicle_region_shape[1]}x${processingInfo.debug_info.vehicle_region_shape[2]}` : 'N/A'}</li>
                                <li>Coordinates: ({processingInfo.debug_info.region_coordinates.y_min}, {processingInfo.debug_info.region_coordinates.x_min}) to 
                                                ({processingInfo.debug_info.region_coordinates.y_max}, {processingInfo.debug_info.region_coordinates.x_max})</li>
                              </ul>
                            </div>
                          )}
                        </div>
                      )}
                      {processingInfo.vehicle_color && (
                        <p className="vehicle-color-caption">
                          Detected Color: {processingInfo.vehicle_color} 
                          {processingInfo.color_confidence && 
                            ` (Confidence: ${(processingInfo.color_confidence * 100).toFixed(1)}%)`}
                        </p>
                      )}
                      {/* After vehicle color section, add vehicle type region */}
                      {intermediateImages.vehicle_region && (
                        <div className="vehicle-type-region">
                          <p><strong>Vehicle Type Detection Region:</strong></p>
                          <img 
                            src={`data:image/jpeg;base64,${intermediateImages.vehicle_region}`} 
                            alt="Vehicle Type Detection Region"
                            onError={(e) => {
                              console.log('Error loading vehicle type region image');
                              e.target.style.display = 'none';
                              e.target.parentNode.innerHTML += '<div class="error-message">Vehicle type region image could not be loaded.</div>';
                            }}
                          />
                          <p className="type-detection-caption">
                            Region used for vehicle type detection: {processingInfo.vehicle_type}
                            {processingInfo.vehicle_type_confidence && 
                              ` (Confidence: ${(processingInfo.vehicle_type_confidence * 100).toFixed(1)}%)`}
                          </p>
                        </div>
                      )}
                    </div>
                  )}
                  
                  {/* Show TensorFlow intermediate stages */}
                  {processingMethod === 'tensorflow' && processingInfo && processingInfo.intermediate_steps && (
                    <div className="intermediate-images">
                      <h3>TensorFlow Pipeline</h3>
                      <p><strong>Original Input Image:</strong></p>
                      <img src={`http://172.20.10.10:8000${processingInfo.intermediate_steps.original}`} alt="Original" />
                      <p><strong>Detection Result:</strong></p>
                      <img 
                        src={`http://172.20.10.10:8000${processingInfo.intermediate_steps.detection}`} 
                        alt="Detection"
                        className="detection-image"
                      />

                      {/* First show plates */}
                      {processingInfo.intermediate_steps.plates && processingInfo.intermediate_steps.plates.map((platePath, index) => (
                        <div key={index} className="detected-plate">
                          <p><strong>Detected Plate {index + 1}:</strong></p>
                          <img src={`http://172.20.10.10:8000${platePath}`} alt={`Plate ${index + 1}`} />
                          {processingInfo.detected_plates && index < processingInfo.detected_plates.length && (
                            <p className="plate-text">Text: {processingInfo.detected_plates[index]}</p>
                          )}
                        </div>
                      ))}
                      
                      {/* Next show vehicle regions if available */}
                      <h4>Vehicle Color Detection:</h4>
                      {processingInfo.intermediate_steps.vehicle_regions && processingInfo.intermediate_steps.vehicle_regions.length > 0 ? (
                        processingInfo.intermediate_steps.vehicle_regions.map((regionPath, index) => (
                          <div key={`region-${index}`} className="detected-vehicle-region">
                            <p><strong>Vehicle Region {index + 1} for Color Detection:</strong></p>
                            <img 
                              src={`http://172.20.10.10:8000${regionPath}`} 
                              alt={`Vehicle Region ${index + 1}`}
                              onError={(e) => {
                                console.log('Error loading vehicle region image');
                                e.target.style.display = 'none';
                                e.target.parentNode.innerHTML += '<div class="error-message">This vehicle region image failed to load.</div>';
                              }}
                            />
                            {processingInfo.vehicle_colors && index < processingInfo.vehicle_colors.length && (
                              <p className="vehicle-color-caption">
                                Detected Color: {processingInfo.vehicle_colors[index]}
                                {processingInfo.color_confidences && index < processingInfo.color_confidences.length && 
                                  ` (Confidence: ${(processingInfo.color_confidences[index] * 100).toFixed(1)}%)`}
                              </p>
                            )}
                          </div>
                        ))
                      ) : (
                        <div className="missing-region">No vehicle region images available for color detection. Color was detected using the full image.</div>
                      )}

                      {/* If vehicle regions not available but we have a vehicle color, show it */}
                      {(!processingInfo.intermediate_steps.vehicle_regions || processingInfo.intermediate_steps.vehicle_regions.length === 0) && processingInfo.vehicle_color && (
                        <p className="vehicle-color-caption">
                          Primary Vehicle Color: {processingInfo.vehicle_color}
                          {processingInfo.color_confidence && 
                            ` (Confidence: ${(processingInfo.color_confidence * 100).toFixed(1)}%)`}
                        </p>
                      )}

                      {/* After vehicle color section in the More window, update vehicle color detection section */}
                      <h4>Vehicle Color Detection:</h4>
                      <div className="vehicle-color-detection-section">
                        {/* First, show the full image used for color detection */}
                        <div className="full-image-color-section">
                          <p><strong>Full Image Used for Color Detection:</strong></p>
                          {processingInfo.intermediate_steps && processingInfo.intermediate_steps.full_image_color ? (
                            <img 
                              src={`http://172.20.10.10:8000${processingInfo.intermediate_steps.full_image_color}`}
                              alt="Full Image Color Detection"
                              className="color-detection-image"
                              onError={(e) => {
                                console.log('Error loading full image color detection');
                                e.target.style.display = 'none';
                                e.target.parentNode.innerHTML += '<div class="error-message">Full image color detection image failed to load.</div>';
                              }}
                            />
                          ) : (
                            <div className="missing-image">Full image color detection image not available.</div>
                          )}
                          <div className="full-image-color-info">
                            <strong>Full Image Color:</strong> {processingInfo.full_image_color || 'Unknown'}
                            {processingInfo.full_image_color_confidence && 
                              <span className="color-confidence"> (Confidence: {(processingInfo.full_image_color_confidence * 100).toFixed(1)}%)</span>}
                          </div>
                          {/* Show color bars for full image */}
                          {processingInfo.color_percentages && Object.keys(processingInfo.color_percentages).length > 0 && (
                            <div className="color-analysis mini">
                              <h5>Full Image Color Distribution:</h5>
                              <div className="color-bars">
                                {Object.entries(processingInfo.color_percentages)
                                  .sort((a, b) => b[1] - a[1])
                                  .map(([color, percentage], index) => (
                                    <div key={index} className="color-bar-row">
                                      <div 
                                        className="color-swatch"
                                        style={{ 
                                          backgroundColor: color,
                                          border: color === 'white' ? '1px solid #ccc' : 'none'
                                        }}
                                      ></div>
                                      <div className="color-label">{color.charAt(0).toUpperCase() + color.slice(1)}</div>
                                      <div className="color-bar-container">
                                        <div 
                                          className="color-bar-fill" 
                                          style={{ 
                                            width: `${Math.min(100, percentage)}%`,
                                            backgroundColor: getColorForBar(color)
                                          }}
                                        ></div>
                                      </div>
                                      <div className="color-percentage">{percentage.toFixed(1)}%</div>
                                    </div>
                                  ))
                                  .slice(0, 3) /* Show top 3 colors only for compact view */
                                }
                              </div>
                            </div>
                          )}
                        </div>
                        
                        {/* Then, show the vehicle region used for color detection if available */}
                        <div className="region-color-section">
                          <p><strong>Vehicle Region Used for Color Detection:</strong></p>
                          {processingInfo.intermediate_steps && processingInfo.intermediate_steps.vehicle_region ? (
                            <img 
                              src={`http://172.20.10.10:8000${processingInfo.intermediate_steps.vehicle_region}`}
                              alt="Vehicle Region Color Detection"
                              className="color-detection-image"
                              onError={(e) => {
                                console.log('Error loading vehicle region color image');
                                e.target.style.display = 'none';
                                e.target.parentNode.innerHTML += '<div class="error-message">Vehicle region color detection image failed to load.</div>';
                              }}
                            />
                          ) : (
                            intermediateImages.vehicle_region ? (
                              <img 
                                src={`data:image/jpeg;base64,${intermediateImages.vehicle_region}`} 
                                alt="Vehicle Region"
                                onError={(e) => {
                                  console.log('Error loading vehicle region image');
                                  e.target.style.display = 'none';
                                  e.target.parentNode.innerHTML += '<div class="error-message">Vehicle region image could not be loaded.</div>';
                                }}
                              />
                            ) : (
                              <div className="missing-image">Vehicle region color detection image not available.</div>
                            )
                          )}
                          <div className="region-color-info">
                            <strong>Region Color:</strong> {processingInfo.region_color || 'Unknown'}
                            {processingInfo.region_color_confidence && 
                              <span className="color-confidence"> (Confidence: {(processingInfo.region_color_confidence * 100).toFixed(1)}%)</span>}
                          </div>
                          {/* Show color bars for region */}
                          {processingInfo.region_color_percentages && Object.keys(processingInfo.region_color_percentages).length > 0 && (
                            <div className="color-analysis mini">
                              <h5>Region Color Distribution:</h5>
                              <div className="color-bars">
                                {Object.entries(processingInfo.region_color_percentages)
                                  .sort((a, b) => b[1] - a[1])
                                  .map(([color, percentage], index) => (
                                    <div key={index} className="color-bar-row">
                                      <div 
                                        className="color-swatch"
                                        style={{ 
                                          backgroundColor: color,
                                          border: color === 'white' ? '1px solid #ccc' : 'none'
                                        }}
                                      ></div>
                                      <div className="color-label">{color.charAt(0).toUpperCase() + color.slice(1)}</div>
                                      <div className="color-bar-container">
                                        <div 
                                          className="color-bar-fill" 
                                          style={{ 
                                            width: `${Math.min(100, percentage)}%`,
                                            backgroundColor: getColorForBar(color)
                                          }}
                                        ></div>
                                      </div>
                                      <div className="color-percentage">{percentage.toFixed(1)}%</div>
                                    </div>
                                  ))
                                  .slice(0, 3) /* Show top 3 colors only for compact view */
                                }
                              </div>
                            </div>
                          )}
                        </div>
                        
                        {/* Finally, show the final vehicle color determination */}
                        <div className="best-color-section">
                          <p><strong>Final Vehicle Color Determination:</strong></p>
                          <div className="best-color">
                            <strong>Best Color Detection:</strong> {processingInfo.vehicle_color}
                            {processingInfo.color_confidence && 
                              <span className="color-confidence"> (Confidence: {(processingInfo.color_confidence * 100).toFixed(1)}%)</span>}
                            {processingInfo.best_color_source && 
                              <span className="source-indicator"> [Source: {processingInfo.best_color_source === 'region' ? 'Vehicle Region' : 'Full Image'}]</span>}
                          </div>
                        </div>
                      </div>
                      
                      {/* After vehicle color detection section, add vehicle type detection section */}
                      <h4>Vehicle Type Detection:</h4>
                      <div className="vehicle-type-detection-section">
                        {/* First, show the full image used for vehicle type detection */}
                        <div className="full-image-type-section">
                          <p><strong>Full Image Used for Type Detection:</strong></p>
                          {processingInfo.intermediate_steps.full_image_type ? (
                            <img 
                              src={`http://172.20.10.10:8000${processingInfo.intermediate_steps.full_image_type}`}
                              alt="Full Image Type Detection"
                              className="type-detection-image"
                              onError={(e) => {
                                console.log('Error loading full image type detection');
                                e.target.style.display = 'none';
                                e.target.parentNode.innerHTML += '<div class="error-message">Full image type detection image failed to load.</div>';
                              }}
                            />
                          ) : (
                            <div className="missing-image">Full image type detection image not available.</div>
                          )}
                          <div className="full-image-detection">
                            <strong>Full Image Detection:</strong> {processingInfo.full_image_type || 'Unknown'}
                            {processingInfo.full_image_type_confidence && 
                              <span className="type-confidence"> (Confidence: {(processingInfo.full_image_type_confidence * 100).toFixed(1)}%)</span>}
                          </div>
                        </div>
                        
                        {/* Then, show the vehicle region used for type detection if available */}
                        <div className="region-type-section">
                          <p><strong>Vehicle Region Used for Type Detection:</strong></p>
                          {processingInfo.intermediate_steps.vehicle_type_region ? (
                            <img 
                              src={`http://172.20.10.10:8000${processingInfo.intermediate_steps.vehicle_type_region}`}
                              alt="Vehicle Type Detection Region"
                              className="type-detection-image"
                              onError={(e) => {
                                console.log('Error loading vehicle type region image');
                                e.target.style.display = 'none';
                                e.target.parentNode.innerHTML += '<div class="error-message">Vehicle region type detection image failed to load.</div>';
                              }}
                            />
                          ) : (
                            <div className="missing-image">Vehicle region type detection image not available.</div>
                          )}
                          <div className="region-detection">
                            <strong>Region Detection:</strong> {processingInfo.region_type || 'Unknown'}
                            {processingInfo.region_type_confidence && 
                              <span className="type-confidence"> (Confidence: {(processingInfo.region_type_confidence * 100).toFixed(1)}%)</span>}
                          </div>
                        </div>
                        
                        {/* Finally, show the final vehicle type determination */}
                        <div className="best-detection-section">
                          <p><strong>Final Vehicle Type Determination:</strong></p>
                          <div className="best-detection">
                            <strong>Best Detection:</strong> {processingInfo.vehicle_type}
                            {processingInfo.vehicle_type_confidence && 
                              <span className="type-confidence"> (Confidence: {(processingInfo.vehicle_type_confidence * 100).toFixed(1)}%)</span>}
                            {processingInfo.best_type_source && 
                              <span className="source-indicator"> [Source: {processingInfo.best_type_source === 'region' ? 'Vehicle Region' : 'Full Image'}]</span>}
                          </div>
                        </div>
                      </div>
                      
                      {/* After the vehicle type detection section, add vehicle make detection section */}
                      <h4>Vehicle Make Detection:</h4>
                      <div className="vehicle-make-detection-section">
                        <div className="full-image-make-section">
                          <p><strong>Full Image Used for Make Detection:</strong></p>
                          {processingInfo && processingInfo.intermediate_steps.full_image_make ? (
                            <img 
                              src={`http://172.20.10.10:8000${processingInfo.intermediate_steps.full_image_make}`}
                              alt="Full Image Make Detection"
                              className="make-detection-image"
                              onError={(e) => {
                                console.log('Error loading full image make detection');
                                e.target.style.display = 'none';
                                e.target.parentNode.innerHTML += '<div class="error-message">Full image make detection image failed to load.</div>';
                              }}
                            />
                          ) : (
                            <div className="missing-image">Full image make detection image not available.</div>
                          )}
                          <div className="full-image-detection">
                            <strong>Full Image Detection:</strong> {processingInfo.full_image_make || 'Unknown'}
                            {processingInfo.full_image_make_confidence && 
                              <span className="make-confidence"> (Confidence: {(processingInfo.full_image_make_confidence * 100).toFixed(1)}%)</span>}
                          </div>
                        </div>
                        
                        {/* Then, show the vehicle region used for make detection if available */}
                        <div className="region-make-section">
                          <p><strong>Vehicle Region Used for Make Detection:</strong></p>
                          {processingInfo && processingInfo.intermediate_steps.vehicle_make_region ? (
                            <img 
                              src={`http://172.20.10.10:8000${processingInfo.intermediate_steps.vehicle_make_region}`}
                              alt="Vehicle Make Detection Region"
                              className="make-detection-image"
                              onError={(e) => {
                                console.log('Error loading vehicle make region image');
                                e.target.style.display = 'none';
                                e.target.parentNode.innerHTML += '<div class="error-message">Vehicle region make detection image failed to load.</div>';
                              }}
                            />
                          ) : (
                            <div className="missing-image">Vehicle region make detection image not available.</div>
                          )}
                          <div className="region-detection">
                            <strong>Region Detection:</strong> {processingInfo.region_make || 'Unknown'}
                            {processingInfo.region_make_confidence && 
                              <span className="make-confidence"> (Confidence: {(processingInfo.region_make_confidence * 100).toFixed(1)}%)</span>}
                          </div>
                        </div>
                        
                        {/* Finally, show the final vehicle make determination */}
                        <div className="best-detection-section">
                          <p><strong>Final Vehicle Make Determination:</strong></p>
                          <div className="best-detection">
                            <strong>Best Detection:</strong> {processingInfo.vehicle_make}
                            {processingInfo.make_confidence && 
                              <span className="make-confidence"> (Confidence: {(processingInfo.make_confidence * 100).toFixed(1)}%)</span>}
                            {processingInfo.best_make_source && 
                              <span className="source-indicator"> [Source: {processingInfo.best_make_source === 'region' ? 'Vehicle Region' : 'Full Image'}]</span>}
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
            <div className="image-container">
              <h3>Annotated {mediaType === 'video' ? 'Video' : 'Image'}:</h3>
              {resultMedia ? (
                mediaType === 'video' ? (
                  <video id="annotatedMedia" src={resultMedia} controls />
                ) : (
                  <img id="annotatedMedia" src={resultMedia} alt="Annotated result" />
                )
              ) : (
                <div className="blank-canvas"></div> // Show blank canvas if result media is not available
              )}
              <button className="more-btn" onClick={() => handleMoreClick('annotated')}>
                More <span className="arrow">{showMoreAnnotated ? '▲' : '▼'}</span>
              </button>
              {showMoreAnnotated && (
                <div className={`more-window ${isExpandedAnnotated ? 'expanded' : ''}`} style={{ backgroundColor: '#1b1924', height: isExpandedAnnotated ? getMediaHeight('annotatedMedia') * 1.3 : getMediaHeight('annotatedMedia'), width: isExpandedAnnotated ? getMediaWidth('annotatedMedia') * 1.3 : getMediaWidth('annotatedMedia') }}>
                  <button className="close-btn" onClick={() => handleCloseClick('annotated')}>×</button>
                  <button className="expand-btn" onClick={() => handleExpandClick('annotated')}>
                    <img src={expandIcon} alt="Expand Icon" style={{ width: '20px', height: '20px', filter: 'invert(100%) sepia(100%) saturate(0%) hue-rotate(60deg) brightness(100%) contrast(100%)' }} />
                  </button>
                  {processingInfo && (
                    <div className="processing-info">
                      {/* Add notification for full image OCR */}
                      {processingInfo.plate_detection_status === 'full_image' && (
                        <div className="full-image-ocr-notice">
                          <p className="warning-text">No license plate detected. Using full image OCR.</p>
                          {processingInfo.original_ocr !== processingInfo.license_plate && (
                            <p className="original-text">Original OCR text: {processingInfo.original_ocr}</p>
                          )}
                        </div>
                      )}
                      
                      <p><strong>Filename:</strong> {processingInfo.filename}</p>
                      <p><strong>License Plate:</strong> {processingInfo.license_plate}</p>
                      
                      {/* Add vehicle color information */}
                      {processingInfo.vehicle_color && (
                        <p className="vehicle-color-info">
                          <strong>Vehicle Color:</strong> {processingInfo.vehicle_color}
                          {processingInfo.color_confidence && (
                            <span className="color-confidence"> (Confidence: {(processingInfo.color_confidence * 100).toFixed(1)}%)</span>
                          )}
                        </p>
                      )}
                      
                      {processingInfo.original_ocr && (
                        <p><strong>Original OCR Detection:</strong> {processingInfo.original_ocr}</p>
                      )}
                      <p><strong>Status:</strong> {processingInfo.status}</p>
                      <p><strong>Result URL:</strong> <a href={processingInfo.result_url} target="_blank" rel="noopener noreferrer">{processingInfo.result_url}</a></p>
                      
                      {/* Enhanced text candidates section */}
                      {processingInfo.text_candidates && (
                        <div className="text-candidates">
                          <h4>Possible License Plate Texts:</h4>
                          <table className="candidates-table">
                            <thead>
                              <tr>
                                <th>Rank</th>
                                <th>Text</th>
                                <th>Confidence</th>
                                <th>Pattern Match</th>
                                <th>Description</th> {/* New column for description */}
                              </tr>
                            </thead>
                            <tbody>
                              {Array.isArray(processingInfo.text_candidates[0]) 
                                ? processingInfo.text_candidates[0].map((candidate, index) => (
                                    <tr key={index} className={
                                      candidate.pattern_name === "Original OCR" ? 'original-candidate' :
                                      index === 0 ? 'best-candidate' : ''
                                    }>
                                      <td>{index + 1}</td>
                                      <td>{candidate.text}</td>
                                      <td>{(candidate.confidence * 100).toFixed(2)}%</td>
                                      <td>{candidate.pattern_match ? '✓' : '✗'}</td>
                                      <td>{candidate.pattern_name || ""}</td>
                                    </tr>
                                  ))
                                : processingInfo.text_candidates.map((candidate, index) => (
                                    <tr key={index} className={
                                      candidate.pattern_name === "Original OCR" ? 'original-candidate' :
                                      index === 0 ? 'best-candidate' : ''
                                    }>
                                      <td>{index + 1}</td>
                                      <td>{candidate.text}</td>
                                      <td>{(candidate.confidence * 100).toFixed(2)}%</td>
                                      <td>{candidate.pattern_match ? '✓' : '✗'}</td>
                                      <td>{candidate.pattern_name || ""}</td>
                                    </tr>
                                  ))
                              }
                            </tbody>
                          </table>
                          
                          {/* New section for per-character confidence */}
                          {processingInfo.text_candidates && processingInfo.text_candidates.length > 0 && (
                            <div className="char-confidence-section">
                              <h4>Character-by-Character Analysis (Best Candidate):</h4>
                              <div className="char-positions-container">
                                {(() => {
                                  // Get the best candidate's character positions data
                                  const bestCandidate = Array.isArray(processingInfo.text_candidates[0]) 
                                    ? processingInfo.text_candidates[0][0] 
                                    : processingInfo.text_candidates[0];
                                
                                  // Check if we have character positions with multiple candidates
                                  const charPositions = bestCandidate?.char_positions || [];
                                
                                  if (charPositions.length === 0 && bestCandidate?.text) {
                                    // If no detailed character data, create a basic display
                                    return bestCandidate.text.split('').map((char, idx) => (
                                      <div key={idx} className="char-position">
                                        <div className={`char-box ${getConfidenceClass(bestCandidate.confidence)}`}>
                                          <div className="char">{char}</div>
                                          <div className="char-conf">{(bestCandidate.confidence * 100).toFixed(0)}%</div>
                                        </div>
                                        <div className="alternatives-spacer"></div>
                                      </div>
                                    ));
                                  }
                                
                                  // Display each character position with alternatives
                                  return charPositions.map((position, idx) => (
                                    <div key={idx} className="char-position">
                                      {/* Main character (best candidate for this position) */}
                                      {position.candidates && position.candidates.length > 0 && (
                                        <>
                                          <div 
                                            className={`char-box ${getConfidenceClass(position.candidates[0].confidence)}`}
                                            title={`Confidence: ${(position.candidates[0].confidence * 100).toFixed(2)}%`}
                                          >
                                            <div className="char">{position.candidates[0].char}</div>
                                            <div className="char-conf">{(position.candidates[0].confidence * 100).toFixed(0)}%</div>
                                          </div>
                                          
                                          {/* Alternative characters */}
                                          <div className="char-alternatives">
                                            {position.candidates.slice(1).map((candidate, altIdx) => (
                                              <div 
                                                key={altIdx}
                                                className={`alt-char-box ${getConfidenceClass(candidate.confidence)}`}
                                                title={`Alternative ${altIdx+1}: ${candidate.char} (${(candidate.confidence * 100).toFixed(2)}%)`}
                                              >
                                                <div className="alt-char">{candidate.char}</div>
                                                <div className="alt-char-conf">{(candidate.confidence * 100).toFixed(0)}%</div>
                                              </div>
                                            ))}
                                          </div>
                                        </>
                                      )}
                                    </div>
                                  ));
                                })()}
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                      
                      {/* Color analysis chart if color_percentages are available */}
                      {processingInfo.color_percentages && (
                        <div className="color-analysis">
                          <h4>Vehicle Color Analysis:</h4>
                          <div className="color-bars">
                            {Object.entries(processingInfo.color_percentages)
                              .sort((a, b) => b[1] - a[1])
                              .map(([color, percentage], index) => (
                                <div key={index} className="color-bar-row">
                                  <div 
                                    className="color-swatch"
                                    style={{ 
                                      backgroundColor: color,
                                      border: color === 'white' ? '1px solid #ccc' : 'none'
                                    }}
                                  ></div>
                                  <div className="color-label">{color.charAt(0).toUpperCase() + color.slice(1)}</div>
                                  <div className="color-bar-container">
                                    <div 
                                      class="color-bar-fill" 
                                      style={{ 
                                        width: `${Math.min(100, percentage)}%`,
                                        backgroundColor: getColorForBar(color)
                                      }}
                                    ></div>
                                  </div>
                                  <div className="color-percentage">{percentage.toFixed(1)}%</div>
                                </div>
                              ))
                              .slice(0, 6) /* Show top 6 colors only */
                            }
                          </div>
                        </div>
                      )}
                      
                      {/* Add Vehicle Type Information */}
                      {processingInfo.vehicle_type && (
                        <div className="vehicle-type-info">
                          <strong>Vehicle Type:</strong> {processingInfo.vehicle_type}
                          {processingInfo.vehicle_type_confidence && (
                            <span className="type-confidence"> 
                              (Confidence: {(processingInfo.vehicle_type_confidence * 100).toFixed(1)}%)
                            </span>
                          )}
                          
                          {/* Show alternative detections if available */}
                          {processingInfo.vehicle_type_alternatives && 
                           processingInfo.vehicle_type_alternatives.length > 0 && (
                            <div className="vehicle-alternatives">
                              <p><strong>Alternative Detections:</strong></p>
                              {processingInfo.vehicle_type_alternatives.map((alt, index) => (
                                <div key={index} className="vehicle-alternative-item">
                                  <span>{alt.type}</span>
                                  <span>{(alt.confidence * 100).toFixed(1)}%</span>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      )}

                      {/* Add Vehicle Orientation Information */}
                      {processingInfo.vehicle_orientation && (
                        <div className="vehicle-orientation-info">
                          <strong>Vehicle Orientation:</strong> {processingInfo.vehicle_orientation}
                          {processingInfo.orientation_confidence && (
                            <span className="orientation-confidence"> 
                              (Confidence: {(processingInfo.orientation_confidence * 100).toFixed(1)}%)
                            </span>
                          )}
                        </div>
                      )}
                      
                      {/* Add Vehicle Make Information after Vehicle Type */}
                      {processingInfo.vehicle_make && (
                        <div className="vehicle-make-info">
                          <strong>Vehicle Make:</strong> {processingInfo.vehicle_make}
                          {processingInfo.make_confidence && (
                            <span className="make-confidence"> 
                              (Confidence: {(processingInfo.make_confidence * 100).toFixed(1)}%)
                            </span>
                          )}
                          
                          {/* Show alternative detections if available */}
                          {processingInfo.make_alternatives && 
                           processingInfo.make_alternatives.length > 0 && (
                            <div className="vehicle-alternatives">
                              <p><strong>Alternative Detections:</strong></p>
                              {processingInfo.make_alternatives.map((alt, index) => (
                                <div key={index} className="vehicle-alternative-item">
                                  <span>{alt.make}</span>
                                  <span>{(alt.confidence * 100).toFixed(1)}%</span>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      )}
                      
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
          {mediaType === 'video' && (
            <button className="play-btn" onClick={handlePlayClick}>Play Both Videos</button>
          )}
          <p>Processing time: {finalTime}</p>
          <div class="dropdown-container">
            <div class="icon-container" onClick={handleReloadClick}>
              <img src={reloadIcon} alt="Option Icon" class="dropdown-icon" />
            </div>
            <select id="options" value={selectedOption} onChange={handleOptionChange}>
              <option value=""></option>
              <option value="opencv">OpenCV</option>
              <option value="tensorflow">TensorFlow</option>
            </select>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="home-page">
      <RealtimeDetection 
        isActive={isRealtimeActive} 
        onClose={handleRealtimeClose}
      />
      <div className="upload-section-container">
        <div className="upload-header">
          <h2>Upload Footage.</h2>
        </div>
        <div className="upload-buttons">
          <button className="upload-btn file" onClick={() => handleFileClick('image')}>
            <img src={fileIcon} alt="File Icon" className="file-icon" />
            <span className="tooltip">Upload Image</span>
          </button>
          <button className="upload-btn video" onClick={() => handleFileClick('video')}>
            <img src={videoIcon} alt="Video Icon" className="video-icon" />
            <span className="tooltip">Upload Video</span>
          </button>
          <button className="upload-btn real-time" onClick={handleRealtimeClick}>
            <img src={realtimeIcon} alt="Real Time Icon" className="real-time-icon" />
            <span className="tooltip">Real-Time Detection</span>
          </button>
        </div>
        <div className="options-container">
          <div className="dropdown-container">
            <div className="icon-container" onClick={handleReloadClick}>
              <img src={reloadIcon} alt="Option Icon" className="dropdown-icon" />
            </div>
            <select id="options" value={selectedOption} onChange={handleOptionChange}>
              <option value=""></option>
              <option value="opencv">OpenCV</option>
              <option value="tensorflow">TensorFlow</option>
            </select>
          </div>
          <div className="checkbox-container">
            <input
              type="checkbox"
              id="low-visibility"
              checked={isLowVisibility}
              onChange={() => setIsLowVisibility(!isLowVisibility)}
            />
            <label htmlFor="low-visibility">Low Visibility/Foggy Image</label>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HomePage;
