import os
import glob
import argparse
import cv2
from datav2.common import extract_features

def get_video_paths():
    parser = argparse.ArgumentParser(description="Prepare data")
    parser.add_argument("--video_folder", type=str, required=True, help="absolute folder path with multiple MP4(s) under it")

    parser.add_argument("--n_fft",      type=int, default=2048, help="n_fft")
    parser.add_argument("--hop_length", type=int, default=512, help="hop_length")
    parser.add_argument("--n_mfcc",     type=int, default=20, help="n_mfcc")
    parser.add_argument("--sr",         type=int, default=44100, help="sr")
    parser.add_argument("--win_size",   type=int, default=64, help="win_size")

    opt = parser.parse_args()

    assert os.path.exists(opt.video_folder), f"video_folder: {opt.video_folder} not exist!"

    video_paths = glob.glob(os.path.join(opt.video_folder, "*.mp4"))
    assert len(video_paths) > 0, f"No valid video under video_folder: {opt.video_folder}!"
    return video_paths, opt


def process_video_into_folder_feature(video_path: str, win_size, sr, n_mfcc, n_fft, hop_length):
    fps = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)
    all_aud_feat, all_pose, all_eye = extract_features(
        video_path,
        fps=fps,
        win_size=win_size,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length)

    # Save features in to folder "*/feature/" as AnnVI-like


if __name__ == '__main__':
    video_paths, opt = get_video_paths()

    for video_path in video_paths:
        process_video_into_folder_feature(video_path, opt.win_size, opt.sr, opt.n_mfcc, opt.n_fft, opt.hop_length)
