import React, { useEffect, useRef, useState } from 'react';
import '../styles/RealtimeDetection.css';

const RealtimeDetection = ({ isActive, onClose }) => {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const wsRef = useRef(null);
    const [error, setError] = useState(null);
    const [isConnected, setIsConnected] = useState(false);
    const frameProcessingRef = useRef(false);
    const [cameras, setCameras] = useState([]);
    const [selectedCamera, setSelectedCamera] = useState('');

    const getCameras = async () => {
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            const videoDevices = devices.filter(device => device.kind === 'videoinput');
            setCameras(videoDevices);
            if (videoDevices.length > 0) {
                setSelectedCamera(videoDevices[0].deviceId);
            }
        } catch (err) {
            console.error('Error getting cameras:', err);
            setError('Unable to get camera list. Please check camera permissions.');
        }
    };

    const setupWebSocket = () => {
        wsRef.current = new WebSocket('ws://172.20.10.10:8000/api/ws/realtime-detection');

        wsRef.current.onopen = () => {
            console.log('WebSocket connected');
            setIsConnected(true);
        };

        wsRef.current.onclose = () => {
            console.log('WebSocket disconnected');
            setIsConnected(false);
        };

        wsRef.current.onerror = (error) => {
            console.error('WebSocket error:', error);
            setError('Connection to detection server failed. Please try again.');
        };

        wsRef.current.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data.success) {
                    drawDetections(data.detections);
                } else if (data.error) {
                    console.error('Detection error:', data.error);
                }
                frameProcessingRef.current = false;
            } catch (err) {
                console.error('Error processing detection result:', err);
                frameProcessingRef.current = false;
            }
        };
    };

    const drawDetections = (detections) => {
        if (!canvasRef.current || !videoRef.current) return;

        const canvas = canvasRef.current;
        const video = videoRef.current;
        const ctx = canvas.getContext('2d');

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        detections.forEach(detection => {
            const [x1, y1, x2, y2] = detection.bbox;
            const text = detection.text || '';

            ctx.strokeStyle = '#00ff00';
            ctx.lineWidth = 2;
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

            ctx.fillStyle = 'rgba(0, 255, 0, 0.3)';
            const textWidth = ctx.measureText(text).width;
            ctx.fillRect(x1, y1 - 20, textWidth + 10, 20);

            ctx.fillStyle = '#ffffff';
            ctx.font = '16px Arial';
            ctx.fillText(text, x1 + 5, y1 - 5);
        });
    };

    const processFrame = async () => {
        if (!videoRef.current || !wsRef.current || !isConnected || frameProcessingRef.current) {
            return;
        }

        const video = videoRef.current;
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        const frame = canvas.toDataURL('image/jpeg', 0.8);
        
        try {
            frameProcessingRef.current = true;
            wsRef.current.send(frame);
        } catch (err) {
            console.error('Error sending frame:', err);
            frameProcessingRef.current = false;
        }
    };

    const startCamera = async () => {
        try {
            if (!selectedCamera) {
                setError('No camera selected. Please select a camera from the dropdown.');
                return;
            }

            const constraints = {
                video: {
                    deviceId: { exact: selectedCamera },
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                }
            };

            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                setError(null);

                videoRef.current.onplaying = () => {
                    const intervalId = setInterval(() => {
                        processFrame();
                    }, 100);
                    videoRef.current.intervalId = intervalId;
                };
            }
        } catch (err) {
            console.error('Error accessing camera:', err);
            
            if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
                setError(
                    'Camera access was denied. Please follow these steps:\n\n' +
                    '1. Check Windows Camera Privacy Settings:\n' +
                    '   • Open Windows Settings\n' +
                    '   • Go to Privacy & Security > Camera\n' +
                    '   • Enable "Camera access" and "Let apps access your camera"\n\n' +
                    '2. Check Browser Settings:\n' +
                    '   • Click the camera icon in your browser\'s address bar\n' +
                    '   • Select "Allow" for camera access\n' +
                    '   • Click the Retry button below\n\n' +
                    '3. If still not working:\n' +
                    '   • Open chrome://settings/content/camera\n' +
                    '   • Add http://172.20.10.10:3000 to allowed sites'
                );
            } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
                setError(
                    'No camera found. Please check:\n\n' +
                    '1. Your camera is properly connected\n' +
                    '2. It\'s not disabled in Device Manager\n' +
                    '3. No other application is using it'
                );
            } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
                setError(
                    'Cannot access camera. Please try:\n\n' +
                    '1. Closing other applications that might be using the camera\n' +
                    '2. Checking if your camera is enabled in Device Manager\n' +
                    '3. Unplugging and reconnecting your camera'
                );
            } else {
                setError(
                    'Unable to access camera. Please try:\n\n' +
                    '1. Go to chrome://settings/content/camera\n' +
                    '2. Add http://172.20.10.10:3000 to allowed sites\n' +
                    '3. Refresh the page and try again'
                );
            }
        }
    };

    const handleCameraChange = (event) => {
        setSelectedCamera(event.target.value);
        // Stop current stream if any
        if (videoRef.current && videoRef.current.srcObject) {
            videoRef.current.srcObject.getTracks().forEach(track => track.stop());
        }
        // Start new stream with selected camera
        startCamera();
    };

    useEffect(() => {
        if (isActive) {
            getCameras();
            setupWebSocket();
        }

        return () => {
            if (wsRef.current) {
                wsRef.current.close();
            }

            if (videoRef.current) {
                if (videoRef.current.intervalId) {
                    clearInterval(videoRef.current.intervalId);
                }
                if (videoRef.current.srcObject) {
                    videoRef.current.srcObject.getTracks().forEach(track => track.stop());
                }
            }
        };
    }, [isActive]);

    const handleClose = () => {
        if (wsRef.current) {
            wsRef.current.close();
        }

        if (videoRef.current) {
            if (videoRef.current.intervalId) {
                clearInterval(videoRef.current.intervalId);
            }
            if (videoRef.current.srcObject) {
                videoRef.current.srcObject.getTracks().forEach(track => track.stop());
            }
        }

        setError(null);
        onClose();
    };

    const handleRetry = () => {
        setError(null);
        if (wsRef.current) {
            wsRef.current.close();
        }
        getCameras();
        setupWebSocket();
        startCamera();
    };

    const renderPlates = (detection) => {
        // Render the plate information with confidence scores
        return (
          <div className="detected-plate">
            <h4>Detected License Plate</h4>
            <p className="plate-text">Text: {detection.text}</p>
            <p className="plate-confidence">Detection Confidence: {(detection.confidence * 100).toFixed(2)}%</p>
            
            {detection.text_candidates && detection.text_candidates.length > 0 && (
              <div>
                <h5>Possible Texts:</h5>
                <table className="candidates-table">
                  <thead>
                    <tr>
                      <th>Rank</th>
                      <th>Text</th>
                      <th>Confidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {detection.text_candidates.slice(0, 10).map((candidate, index) => (
                      <tr key={index} className={index === 0 ? 'best-candidate' : ''}>
                        <td>{index + 1}</td>
                        <td>{candidate.text}</td>
                        <td>{(candidate.confidence * 100).toFixed(2)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        );
      };

    return (
        <div className={`realtime-container ${isActive ? 'active' : ''}`}>
            {error ? (
                <div className="error-message">
                    <p>{error}</p>
                    <button className="retry-button" onClick={handleRetry}>
                        Retry Camera Access
                    </button>
                </div>
            ) : (
                <div className="video-container">
                    <div className="camera-select">
                        <select 
                            value={selectedCamera} 
                            onChange={handleCameraChange}
                            className="camera-dropdown"
                        >
                            {cameras.map(camera => (
                                <option key={camera.deviceId} value={camera.deviceId}>
                                    {camera.label || `Camera ${cameras.indexOf(camera) + 1}`}
                                </option>
                            ))}
                        </select>
                    </div>
                    <video
                        ref={videoRef}
                        className="detection-canvas"
                        autoPlay
                        playsInline
                        muted
                    />
                    <canvas
                        ref={canvasRef}
                        className="detection-overlay"
                    />
                    {!isConnected && (
                        <div className="connection-status">
                            Connecting to detection server...
                        </div>
                    )}
                </div>
            )}
            <button className="close-button" onClick={handleClose}>
                Close
            </button>
        </div>
    );
};

export default RealtimeDetection;