import os
import glob
import argparse

def get_video_paths():
    parser = argparse.ArgumentParser(description="Prepare data")
    parser.add_argument("--video_folder", type=str, required=True, help="absolute folder path with multiple MP4(s) under it")
    opt = parser.parse_args()

    assert os.path.exists(opt.video_folder), f"video_folder: {opt.video_folder} not exist!"

    video_paths = glob.glob(os.path.join(opt.video_folder, "*.mp4"))
    assert len(video_paths) > 0, f"No valid video under video_folder: {opt.video_folder}!"
    return video_paths


def process_video_into_folder_feature(video_path: str):
    



if __name__ == '__main__':
    video_paths = get_video_paths()

    for video_path in video_paths:
        process_video_into_folder_feature(video_path)
