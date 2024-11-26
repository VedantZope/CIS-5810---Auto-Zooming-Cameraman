
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import time
import tempfile
import numpy as np
import cv2
import streamlit as st
import os 
import subprocess
import json

# Get the parent directory of 'main'
current_dir = Path(__file__).parent.parent
sys.path.append(str(current_dir))

from filters.kuwahara import kuwahara_process_video
import helper
import shutil

# Import object detection model - YOLOv8 
from Object_Detection.YOLO_v8_model_helper import *

def init_session_state():
    """Initialize session state variables"""
    if 'video_file' not in st.session_state:
        st.session_state.video_file = None
    if 'segmentation_done' not in st.session_state:
        st.session_state.segmentation_done = False
    if 'heatmap_done' not in st.session_state:
        st.session_state.heatmap_done = False
    if 'filter_done' not in st.session_state:
        st.session_state.filter_done = False
    if 'processing_started' not in st.session_state:
        st.session_state.processing_started = False

def upload_video():
    """Handle video upload"""
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov"]
    )
    if uploaded_file is not None:
        st.session_state.video_file = uploaded_file

        # save uploaded video to disc
        if "UI_videos" not in os.listdir():
            os.mkdir("./UI_videos")

        temp_file_to_save = './UI_videos/input.mp4'
        helper.write_bytesio_to_file(temp_file_to_save, uploaded_file)
        st.session_state.input_video_file = temp_file_to_save
        return True
    return False

def run_segmentation(model_type, video_file):
    """Simulate segmentation processing"""

    tfile_model = tempfile.NamedTemporaryFile(delete=False)
    tfile_model.write(video_file.read())

    outputfile = f"./UI_videos/model_{model_type}_output.mp4"

    with st.spinner("Running segmentation..."):

        if model_type =="YOLOv8":
            print("YOLO")

            frame_detections = YOLO(INPUT_VIDEO=st.session_state.input_video_file, OUTPUT_VIDEO=outputfile)

            st.session_state.yolo_frame_detections = frame_detections

        st.success("Segmentation completed!")
    
    st.session_state.segmentation_done = True

    #Extract INPUT Frames
    path = "./UI_videos/input_frames/"
    helper.make_path(path=path)
    helper.extract_frames(video_path=st.session_state.input_video_file,output_dir=path)

    # Extract SEGMENTATION frames if needed
    path = "./UI_videos/model_frames/"
    helper.make_path(path=path)
    helper.extract_frames(video_path=outputfile,output_dir=path)

    convertedVideo_model = f"./UI_videos/model_{model_type}_output_h264.mp4"
    helper.convert_video_h264(input_file=outputfile, output_file=convertedVideo_model)

    st.session_state.modelh264_video_file = convertedVideo_model

    if st.session_state.debug:
        st.video(convertedVideo_model)

def generate_heatmap(model_type):
    """Generate and display heatmap"""
    
    outputfile = f"./UI_videos/model_{model_type}_merged.mp4"

    with st.spinner("Generating heatmap..."):
        if model_type=='YOLOv8':
            WEIGHTS = {'person': 10, 'Basketball': 50}
            blurred_heatmaps = generate_heatmap_video(frame_detections= st.session_state.yolo_frame_detections,video_path= st.session_state.modelh264_video_file,
                                    output_path=outputfile, return_heatmaps = True, weight_mapping=WEIGHTS)
            
            # with open('merged_frames_heatmap_yolov8.txt', 'w') as convert_file: 
            #     convert_file.write(json.dumps(heatmaps))

            # np.save('merged_frames_heatmap_yolov8.npy', blurred_heatmaps)
            
            # Extract MERGED frames if needed
            path = "./UI_videos/model_merged_frames/"
            helper.make_path(path=path)
            helper.extract_frames(video_path=outputfile,output_dir=path)

            # Store frame names in memory 
            frame_names = [
                p for p in os.listdir(path)
                if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
            ]
        
        # Blur the merged maps now 
        # blur_path = "./UI_videos/model_blurred_frames/"
        # helper.make_path(path=blur_path)
        # helper.blur_maps(frames=frame_names,input_video_dir=path, object_masks=heatmaps,
        #                  iters = 1,save_frames=True, output_dir= blur_path)
        
        # output_blurred_file = f"./UI_videos/model_{model_type}_blurred.mp4"
        # # Stitch frames to video
        # helper.stitch_frames_to_video(
        #     frames = "", 
        #     frames_dir=blur_path, 
        #     from_dir=True,
        #     output_video_path=output_blurred_file
        # )

    convertedVideo_blurred = f"./UI_videos/model_{model_type}_blurred_h264.mp4"
    helper.convert_video_h264(input_file=outputfile, output_file=convertedVideo_blurred)
    st.session_state.modelh264_video_file = convertedVideo_blurred

    if st.session_state.debug:
        st.video(convertedVideo_blurred)

    st.session_state.heatmap_done = True


def apply_filter(filter_type, video_file):
    """Apply selected filter to video"""

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    outputfile = "./UI_videos/filter_output.mp4"

    with st.spinner(f"Applying {filter_type} filter..."):
        # Process the video
        if filter_type=="Kuwahara":
            kuwahara_process_video(input_path=st.session_state.modelh264_video_file,
                                                output_path = outputfile, kuwahara_param = st.session_state.kuwahara_param)
        else:
            st.text("No filter selected.")

        st.success(f"{filter_type} filter applied! - Converting to Appropriate Codec.")
        st.session_state.filter_done = True
    
    convertedVideo = "./UI_videos/filter_output_h264.mp4"
    helper.convert_video_h264(input_file=outputfile, output_file=convertedVideo)

    # Extract frames for analysis later

    path = "./UI_videos/filter_frames/"
    helper.make_path(path=path)
    helper.extract_frames(video_path=convertedVideo,output_dir=path)

    if st.session_state.debug:
        st.video(convertedVideo)
    
def show_video_details(video_file):
    """Display video metadata"""
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    video = cv2.VideoCapture(tfile.name)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps

    if st.session_state.debug:
        st.subheader("Video Details")
        col1, col2, col3 = st.columns(3)
        col1.metric("FPS", f"{fps:.2f}")
        col2.metric("Frames", frame_count)
        col3.metric("Duration (s)", f"{duration:.2f}")
        st.video(video_file)

def hometab():
    if st.session_state.video_file is not None:
        st.header("Processing Pipeline")

        # Show video details
        show_video_details(st.session_state.video_file)

        if st.session_state.processing_started:
            # Run segmentation
            st.subheader("Step 1: Segmentation")
            if not st.session_state.segmentation_done:
                run_segmentation(st.session_state.segmentation_model, st.session_state.video_file)

            # Generate heatmap
            if st.session_state.segmentation_done:
                st.subheader("Step 2: Heatmap Generation")
                if not st.session_state.heatmap_done:
                    generate_heatmap(st.session_state.segmentation_model)

            # Apply filter if selected
            if st.session_state.heatmap_done and st.session_state.filter_type != "None":
                st.subheader("Step 3: Applying Filter")
                if not st.session_state.filter_done:
                    apply_filter(st.session_state.filter_type, st.session_state.video_file)
                    # st.session_state.filtered_video = output_video

            # Final output
            if st.session_state.heatmap_done:
                st.subheader("Final Output")
                # st.download_button(
                #     label="Download Processed Video",
                #     data=st.session_state..getvalue(),
                #     file_name="processed_video.mp4",
                #     mime="video/mp4"
                # )
    else:
        st.info("Please upload a video file to begin processing")

def frameAnalysisTab():
    #TODO - Add plotly plot here to compare frame by frame  
    st.header("Content in Second Tab") 
    st.write("This is the content of the second tab")
    # Add different components
    st.slider("Select a value", 0, 50)

def sidebar():
    st.header("Processing Steps")

    # Step 1: Video Upload
    st.subheader("1. Upload Video")
    if upload_video():
        st.success("Video uploaded!")

    # Debug 
    debug = st.checkbox(label="Debug")
    st.session_state.debug = debug

    # Step 2: Segmentation Settings
    st.subheader("2. Segmentation Settings")
    segmentation_model = st.selectbox(
        "Select Segmentation Model",
        ["YOLOv8", "Model B", "Model C"]
    )
    st.session_state.segmentation_model = segmentation_model

    # Step 3: Heatmap Settings
    st.subheader("3. Heatmap Settings")
    heatmap_type = st.selectbox(
        "Select Heatmap Type",
        ["Type 1", "Type 2", "Type 3"]
    )
    st.session_state.heatmap_type = heatmap_type

    # Step 4: Filter Settings
    st.subheader("4. Filter Settings (Optional)")
    filter_type = st.selectbox(
        "Select Filter",
        ["None", "Blur", "Sharpen", "Grayscale", "Kuwahara"]
    )
    st.session_state.filter_type = filter_type
    if filter_type=="Kuwahara":
        kuwahara_param = st.number_input(label="Kuwahara radius", min_value=1, max_value=10, step=1, format="%i")
        st.session_state.kuwahara_param = kuwahara_param

    # Start Processing Button
    if st.button("Start Processing") and st.session_state.video_file is not None:
        st.session_state.processing_started = True

def main():
    st.title("Computer Vision Final Project Group 30 Dashboard")
    init_session_state()

    # Tabs
    home_tab, frame_analysis_tab = st.tabs(["Home", "Analyze Frames"])

    # Add content to first tab
    with home_tab:
        hometab()

    # Add content to second tab
    with frame_analysis_tab:
        frameAnalysisTab()

    # Left sidebar for step selection
    with st.sidebar:
        sidebar()

if __name__ == "__main__":
    main()
