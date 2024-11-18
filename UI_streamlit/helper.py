import subprocess
import cv2
from stqdm import stqdm
import os
import shutil

# func to save BytesIO on a drive
def write_bytesio_to_file(filename, bytesio):
    """
    Write the contents of the given BytesIO to a file.
    Creates the file or overwrites the file if it does
    not exist yet. 
    """
    with open(filename, "wb") as outfile:
        # Copy the BytesIO stream to the output file
        outfile.write(bytesio.getbuffer())

def make_path(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def convert_video_h264(input_file, output_file):
    ## Reencodes video to H264 using ffmpeg
    ##  It calls ffmpeg back in a terminal so it fill fail without ffmpeg installed
    ##  ... and will probably fail in streamlit cloud
    subprocess.call(args=f"ffmpeg -y -i {input_file} -c:v libx264 {output_file}".split(" "))

# Get Video data
def extract_frames(video_path, output_dir):
    # Check if the output directory exists, if not create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Capture video
    video = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not video.isOpened():
        print(f"Error opening video file: {video_path}")
        return
    
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    # while True:
    for _ in stqdm(range(total_frames), desc="Extracting frames"):
        # Read a frame
        ret, frame = video.read()

        # Break the loop if no more frames
        if not ret:
            break

         # Save the frame as an image file
        resized_frame = cv2.resize(frame, (640,480))
        frame_path = os.path.join(output_dir, f"{frame_count:04d}.jpg")
        cv2.imwrite(frame_path, resized_frame)
        frame_count+=1

    # Release the video capture object
    video.release()
    print(f"Extracted {frame_count} frames and saved in {output_dir}")

 