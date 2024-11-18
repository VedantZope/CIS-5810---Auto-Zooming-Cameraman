
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

# Get the parent directory of 'main'
current_dir = Path(__file__).parent.parent
sys.path.append(str(current_dir))

from filters.kuwahara import kuwahara_process_video
import helper
import shutil

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


def run_segmentation():
    """Simulate segmentation processing"""
    with st.spinner("Running segmentation..."):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.02)
            progress_bar.progress(i + 1)
        st.success("Segmentation completed!")

        # Display dummy segmentation result
        fig, ax = plt.subplots()
        dummy_seg = np.random.rand(100, 100)
        sns.heatmap(dummy_seg, ax=ax)
    
    if st.session_state.debug:
        st.pyplot(fig)
    st.session_state.segmentation_done = True



    #Extract INPUT Frames
    path = "./UI_videos/input_frames/"
    helper.make_path(path=path)
    helper.extract_frames(video_path=st.session_state.input_video_file,output_dir=path)

    #TODO Extract SEGMENTATION frames if needed


def generate_heatmap():
    """Generate and display heatmap"""
    with st.spinner("Generating heatmap..."):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)

        fig, ax = plt.subplots(figsize=(10, 6))
        data = np.random.rand(20, 20)
        sns.heatmap(data, cmap='viridis', ax=ax)
    
    if st.session_state.debug:
        st.pyplot(fig)
    st.success("Heatmap generated!")
    st.session_state.heatmap_done = True


def apply_filter(filter_type, video_file):
    """Apply selected filter to video"""

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    outputfile = "./UI_videos/filter_output.mp4"

    with st.spinner(f"Applying {filter_type} filter..."):
        # Process the video
        if filter_type=="Kuwahara":
            kuwahara_process_video(input_path=st.session_state.input_video_file,
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
                run_segmentation()

            # Generate heatmap
            if st.session_state.segmentation_done:
                st.subheader("Step 2: Heatmap Generation")
                if not st.session_state.heatmap_done:
                    generate_heatmap()

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
        ["Model A", "Model B", "Model C"]
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
