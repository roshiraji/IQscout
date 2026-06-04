"""
Read MP4 video metadata with OpenCV.

MVP stage: opens a video file and prints basic properties.
Does not run YOLO or tracking yet.
"""

import sys
from pathlib import Path

import cv2


def main() -> None:
    # 1. Parse the video path from the command line
    if len(sys.argv) < 2:
        print("Usage: python src/process_video.py <path_to_video.mp4>")
        sys.exit(1)

    video_path = Path(sys.argv[1])

    # 2. Check that the file exists before opening it
    if not video_path.is_file():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    print(f"Opening video: {video_path.resolve()}")

    # 3. Open the video with OpenCV
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        sys.exit(1)

    # 4. Read basic metadata from the video stream
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 5. Print the metadata (useful for debugging and later pipeline steps)
    print("\n--- Video metadata ---")
    print(f"  FPS:         {fps}")
    print(f"  Width:       {width} px")
    print(f"  Height:      {height} px")
    print(f"  Frame count: {frame_count}")
    print("----------------------\n")

    # 6. Release the video capture object
    cap.release()
    print("Done. (YOLO detection not enabled in this MVP step.)")


if __name__ == "__main__":
    main()
