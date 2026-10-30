import cv2
import os


# List of video indices to process
video_indices = [0, 3]

# List of frames to extract from each video
frames_to_extract = [10, 20, 30]

video_directory = "videos"
video_list = [f"output{index}.avi" for index in video_indices]

# Output directory for extracted frames
output_dir = "extracted_frames"
os.makedirs(output_dir, exist_ok=True)

for video_filename in video_list:
    video_path = os.path.join(video_directory, video_filename)

    # Check if the video file exists
    if not os.path.isfile(video_path):
        print(f"Video file {video_path} does not exist.")
        continue

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        continue

    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {video_filename}, Total Frames: {total_frames}")

    for frame_num in frames_to_extract:
        if frame_num >= total_frames:
            print(
                f"Frame {frame_num} exceeds total frames {total_frames} in video {video_filename}"
            )
            continue

        # Set the frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

        # Read the frame
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame {frame_num} from video {video_filename}")
            continue

        # Construct output filename
        video_base = os.path.splitext(video_filename)[0]
        output_filename = f"{video_base}_frame{frame_num}.png"
        output_path = os.path.join(output_dir, output_filename)

        # Save the frame as a PNG image with maximum quality
        # Setting compression to 0 for maximum quality
        cv2.imwrite(output_path, frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

        print(f"Saved frame {frame_num} from {video_filename} as {output_filename}")

    cap.release()
