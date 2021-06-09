import os
import glob
import argparse
import cv2
import torch
from datav2.common import extract_features

def get_video_paths():
    parser = argparse.ArgumentParser(description="Prepare data")
    parser.add_argument("--video_folder", type=str, required=True,  help="absolute folder path with multiple MP4(s) under it")
    parser.add_argument("--device",       type=int, default=1,      help="GPU device index")

    parser.add_argument("--n_fft",        type=int, default=2048,     help="n_fft")
    parser.add_argument("--hop_length",   type=int, default=512,      help="hop_length")
    parser.add_argument("--n_mfcc",       type=int, default=20,       help="n_mfcc")
    parser.add_argument("--sr",           type=int, default=44100,    help="sr")
    parser.add_argument("--win_size",     type=int, default=64,       help="win_size")

    opt = parser.parse_args()

    # Check opt.video_folder available
    assert os.path.exists(opt.video_folder), f"video_folder: {opt.video_folder} not exist!"

    # Get video paths
    video_paths = glob.glob(os.path.join(opt.video_folder, "*.mp4"))
    assert len(video_paths) > 0, f"No valid video under video_folder: {opt.video_folder}!"

    # Create image/, feature/ under opt.video_folder
    image_folder = os.path.join(opt.video_folder, "image")
    feature_folder = os.path.join(opt.video_folder, "feature")
    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(feature_folder, exist_ok=True)

    data_paths = []
    for video_path in video_paths:
        video_image_folder = os.path.join(image_folder, os.path.splitext(os.path.basename(video_path))[0])
        video_feature_folder = os.path.join(feature_folder, os.path.splitext(os.path.basename(video_path))[0])
        os.makedirs(video_image_folder, exist_ok=True)
        os.makedirs(video_feature_folder, exist_ok=True)

        data_paths.append({
            "video_path": video_path,
            "video_image_folder": video_image_folder,
            "video_feature_folder": video_feature_folder})
    return data_paths, opt, feature_folder


def process_video_into_folder_feature(data_path: dict, win_size, sr, n_mfcc, n_fft, hop_length, img_size=256):
    video_path = data_path["video_path"]
    video_image_folder = data_path["video_image_folder"]
    video_feature_folder = data_path["video_feature_folder"]

    fps = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)
    pil_images, all_aud_feat, all_pose, all_eye = extract_features(
        video_path,
        fps=fps,
        win_size=win_size,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length)

    # Save split images
    for idx, image in enumerate(pil_images):
        image.save(os.path.join(video_image_folder, f"{idx}.jpg"))
        
    # # Save features in to folder "*/feature/" as AnnVI-like
    # # train dataset
    # img_paths_train = [img_paths[i] for i in index[:length_train]]
    # audio_features_train = [audio_features[i] for i in index[:length_train]]
    # lands_train = [lands[i] for i in index[:length_train]]
    # poses_train = [poses[i] for i in index[:length_train]]
    # eyes_train = [eyes[i] for i in index[:length_train]]
    # # test dataset
    # img_paths_test = [img_paths[i] for i in index[length_train:]]
    # audio_features_test = [audio_features[i] for i in index[length_train:]]
    # lands_test = [lands[i] for i in index[length_train:]]
    # poses_test = [poses[i] for i in index[length_train:]]
    # eyes_test = [eyes[i] for i in index[length_train:]]
    #
    # save_data_train = {'img_paths': img_paths_train, 'audio_features': audio_features_train, 'lands': lands_train,
    #                    'poses': poses_train, 'eyes': eyes_train}
    # torch.save(save_data_train, os.path.join(video_feature_folder, '{}_train.t7'.format(img_size)))
    #
    # save_data_test = {'img_paths': img_paths_test, 'audio_features': audio_features_test, 'lands': lands_test,
    #                   'poses': poses_test, 'eyes': eyes_test}
    # torch.save(save_data_test, os.path.join(video_feature_folder, '{}_test.t7'.format(img_size)))


if __name__ == '__main__':
    data_paths, opt, final_feature_folder = get_video_paths()

    for data_path in data_paths:
        process_video_into_folder_feature(data_path, opt.win_size, opt.sr, opt.n_mfcc, opt.n_fft, opt.hop_length)

    print(f"To run train*.py, please additionally assign:\n  --data_root {final_feature_folder}")