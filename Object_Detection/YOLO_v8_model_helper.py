import cv2
import numpy as np
import os
import sys
from ultralytics import YOLO
from Object_Detection.detect import *
from Object_Detection.utils import *
from stqdm import stqdm

# Paths to your custom YOLOv8 models
MODEL_PATH = 'Object_Detection/Models/bb_ball.pt'
PERSON_MODEL_PATH = 'Object_Detection/Models/person.pt'
# INPUT_VIDEO = 'NBA Game 0021800013.mp4'  # Replace with your input video path
# OUTPUT_VIDEO = 'output_video_obj.mp4'  # Replace with desired output path

# Confidence thresholds
PERSON_CONFIDENCE_THRESHOLD = 0.5
BALL_CONFIDENCE_THRESHOLD = 0.3
NMS_THRESHOLD = 0.4

# Device for model inference
DEVICE = 'mps'

# Color definitions for drawing
COLORS = {
    'person': (255, 0, 0),          # Blue
    'Basketball': (0, 255, 0),      # Green
    'Made-Basket': (0, 0, 255),     # Red
}

WEIGHTS = {'person': 10, 'Basketball': 100}


import cv2
import numpy as np
import os

def generate_heatmap_video(frame_detections, video_path, output_path, weight_mapping, return_heatmaps=False):
    """
    Generates a heatmap video based on frame detections with specified weights for each class.
    
    Parameters:
    - frame_detections (list): List of dictionaries containing 'frame_number' and 'detections'.
                                Format: [{'frame_number': 1, 'detections': [{'bbox': [x1, y1, x2, y2], 'label': 'person', 'confidence': 0.85}, ...]}, ...]
    - video_path (str): Path to the input video file.
    - output_path (str): Path to save the output heatmap video.
    - weight_mapping (dict): Dictionary mapping class labels to their corresponding weights.
                             Example: {'person': 1, 'sports ball': 10}
    - return_heatmaps (bool): If True, returns a list of heatmaps for each frame.
    
    Returns:
    - per_frame_heatmaps (list, optional): List of heatmaps for each frame.
                                           Each heatmap is a 2D numpy array.
    """
    # Check if input video exists
    if not os.path.exists(video_path):
        print(f"Input video file not found: {video_path}")
        return
    
    # Open the original video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video Properties:\n- Resolution: {frame_width}x{frame_height}\n- FPS: {fps}\n- Total Frames: {total_frames}")
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can choose other codecs like 'XVID'
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    print("Starting heatmap video creation...")
    
    # Initialize list to store per-frame heatmaps if needed
    per_frame_heatmaps = [] if return_heatmaps else None
    
    # Ensure frame_detections are sorted by frame_number
    frame_detections_sorted = sorted(frame_detections, key=lambda x: x['frame_number'])
    detections_index = 0
    num_detections = len(frame_detections_sorted)
    
    for frame_number in range(1, total_frames + 1):
        ret, frame = cap.read()
        if not ret:
            print(f"End of video reached at frame {frame_number}.")
            break
        
        # Check if current frame has detections
        if detections_index < num_detections and frame_detections_sorted[detections_index]['frame_number'] == frame_number:
            detections = frame_detections_sorted[detections_index]['detections']
            detections_index += 1
            print(f"Frame {frame_number}: Processing {len(detections)} detections.")
        else:
            detections = []
            print(f"Frame {frame_number}: No detections.")
        
        # Initialize a new heatmap for the current frame
        frame_heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)
        
        # Update heatmap with current frame's detections
        for det in detections:
            label = det['label']
            bbox = det['bbox']  # [x1, y1, x2, y2]
            weight = weight_mapping.get(label, 1)  # Default weight is 1 if label not in mapping
            
            # Extract bbox coordinates
            x1, y1, x2, y2 = bbox
            # Ensure coordinates are within frame bounds and integers
            x1 = max(0, min(int(x1), frame_width - 1))
            y1 = max(0, min(int(y1), frame_height - 1))
            x2 = max(0, min(int(x2), frame_width - 1))
            y2 = max(0, min(int(y2), frame_height - 1))
            
            # Add weight to the heatmap region
            frame_heatmap[y1:y2, x1:x2] += weight
        
        # Normalize heatmap to the range [0, 255]
        normalized_heatmap = np.clip(frame_heatmap, 0, 255).astype(np.uint8)
        
        # Apply color map to the heatmap
        heatmap_color = cv2.applyColorMap(normalized_heatmap, cv2.COLORMAP_HOT)
        
        # Optional: Store the heatmap for this frame
        if return_heatmaps:
            per_frame_heatmaps.append(frame_heatmap.copy())
        
        # Overlay the heatmap onto the original frame
        alpha = 0.3  # Transparency for the original frame
        beta = 0.7   # Transparency for the heatmap
        overlay = cv2.addWeighted(frame, alpha, heatmap_color, beta, 0)
        
        # Write the overlaid frame to the output video
        out.write(overlay)
        
        # Display the frame with heatmap (optional)
        # cv2.imshow('Heatmap Video', overlay)
        
        # Press 'q' to exit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Early exit triggered.")
            break
        
        # Print progress every 30 frames
        if frame_number % 30 == 0:
            print(f"Processed {frame_number}/{total_frames} frames.")
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Heatmap video creation complete.")
    print(f"Output saved to: {output_path}")
    
    # Return heatmaps if requested
    if return_heatmaps:
        return per_frame_heatmaps


def YOLO(INPUT_VIDEO, OUTPUT_VIDEO, MODEL_PATH = MODEL_PATH, PERSON_MODEL_PATH = PERSON_MODEL_PATH,
          PERSON_CONFIDENCE_THRESHOLD = PERSON_CONFIDENCE_THRESHOLD, BALL_CONFIDENCE_THRESHOLD = BALL_CONFIDENCE_THRESHOLD, NMS_THRESHOLD = NMS_THRESHOLD, 
          DEVICE = DEVICE, COLORS = COLORS, WEIGHTS = WEIGHTS):
    # Check if model files exist
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found: {MODEL_PATH}")
        return
    if not os.path.exists(PERSON_MODEL_PATH):
        print(f"Person model file not found: {PERSON_MODEL_PATH}")
        return
    if not os.path.exists(INPUT_VIDEO):
        print(f"Input video not found: {INPUT_VIDEO}")
        return

    # Load YOLOv8 models
    print("Loading YOLOv8 models...")
    person_model = load_yolo_model(PERSON_MODEL_PATH)
    basketball_model = load_yolo_model(MODEL_PATH)
    print("Models loaded successfully.")

    # Models dictionary
    models = {
        'person_model': person_model,
        'basketball_model': basketball_model
    }

    # Open video file
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"Error opening video file: {INPUT_VIDEO}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' codec for MP4 format
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (frame_width, frame_height))

    print("Processing video...")

    frame_count = 0

    # Initialize state
    state = {
        'tracker': None,
        'tracking': False,
        'bbox': None,
        'last_class_id': None,
        'tracker_type': 'CSRT',  # You can change to 'KCF', 'MOSSE', etc.
    }

    # Initialize variable to collect labels and bounding boxes for every frame
    frame_detections = []

    # while True:
    for _ in stqdm(range(total_frames), desc="Detecting Objects video"):
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Process the frame
        frame, state, detections = process_frame(frame, models, state, frame_count, DEVICE, COLORS, PERSON_CONFIDENCE_THRESHOLD, BALL_CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

        # Collect labels and bounding boxes
        frame_detections.append({
            'frame_number': frame_count,
            'detections': detections  # This can be a list of detection dicts
        })

        # Display the resulting frame (optional)
        # cv2.imshow('Basketball Detection and Tracking', frame)

        # Write the frame to the output video
        out.write(frame)

        # Press 'q' to exit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")

    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Processing complete. Output saved to:", OUTPUT_VIDEO)

    # Return or save the frame_detections variable as needed
    return frame_detections

# if __name__ == "__main__":
#     frame_detections = np.load('frame_detections.npy', allow_pickle=True)
#     # Optional: Print the detections per frame
#     print(frame_detections)
#     generate_heatmap_video(frame_detections, INPUT_VIDEO, 'output_heatmap_obj.mp4', WEIGHTS, return_heatmaps=False)

