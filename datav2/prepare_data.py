import os
import glob
from tqdm import tqdm
import argparse
import numpy as np
import cv2
from PIL import Image
import torch
import random

from incubator.APB2FaceV2.datav2.common import split_video, extract_features
from incubator.HRNet.inference import init as HRNet_init
from incubator.HRNet.inference import inference as HRNet_inference

ANNVI_LANDMARKS_POINTS = 106

flatten = lambda t: [item for sublist in t for item in sublist]

def get_video_paths():
    parser = argparse.ArgumentParser(description="Prepare data")
    parser.add_argument("--video_folder", type=str, required=True,  help="absolute folder path with multiple MP4(s) under it")

    # HRNet model
    parser.add_argument("--rf_model_path", type=str, required=True, help="path of Resnet50_Final.pth")
    parser.add_argument("--lm_model_path", type=str, required=True, help="path of HR18-WFLW.pth")

    parser.add_argument("--device",       type=int, default=0,      help="GPU device index")
    parser.add_argument("--n_fft",        type=int, default=2048,   help="n_fft")
    parser.add_argument("--hop_length",   type=int, default=512,    help="hop_length")
    parser.add_argument("--n_mfcc",       type=int, default=20,     help="n_mfcc")
    parser.add_argument("--sr",           type=int, default=44100,  help="sr")
    parser.add_argument("--win_size",     type=int, default=64,     help="win_size")

    opt = parser.parse_args()

    # Set cuda device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.device)

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


def init_HRNet(rf_model_path, lm_model_path, points=ANNVI_LANDMARKS_POINTS):
    rf_detector, lm_model = HRNet_init(rf_model_path, lm_model_path, points)
    def inference_HRNet(pil_img, rf_detector=rf_detector, lm_model=lm_model):
        return HRNet_inference(rf_detector, lm_model, pil_img, points)
    return inference_HRNet


def crop_pil_image_face(pil_image, landmarks_106, scale: float=1.2, target_size: int=512):
    assert len(landmarks_106) == 106, "landmarks must be 106 points"

    width, height = pil_image.size

    np_landmarks_106 = np.array(landmarks_106)
    x_max, y_max = np.max(np_landmarks_106, axis=0)
    x_min, y_min = np.min(np_landmarks_106, axis=0)

    x_center, y_center = (x_max + x_min) // 2, (y_max + y_min) // 2
    w_half, h_half = x_max - x_center, y_max - y_center
    w_half = h_half = int(max(w_half, h_half) * scale)

    left, top, right, bottom = max(x_center - w_half, 0), \
                               max(y_center - h_half, 0), \
                               min(x_center + w_half, width), \
                               min(y_center + h_half, height)
    # TODO: Check to square

    crop_image = pil_image.crop((left, top, right, bottom)).resize((target_size, target_size))
    return crop_image


def process_video_into_folder_feature(data_path: dict, landmark_inference_func, win_size, sr, n_mfcc, n_fft, hop_length, img_size=256):
    video_path = data_path["video_path"]
    video_image_folder = data_path["video_image_folder"]
    video_feature_folder = data_path["video_feature_folder"]

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    print("* Start split frames")
    audio_path, image_paths = split_video(video_path, video_image_folder)

    print("* Start extract features")
    audio_features, poses, eyes = extract_features(
        audio_path, image_paths,
        fps=fps,
        win_size=win_size,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length)

    # Save split images
    print("* Start extract landmarks and save cropped images")
    crop_image_folder = f"{img_size}_image_crop"
    os.makedirs(os.path.join(video_feature_folder, crop_image_folder), exist_ok=True)
    img_paths = []
    lands = []
    for idx, image_path in tqdm(enumerate(image_paths)):
        crop_image_path = f"{crop_image_folder}/{idx}.jpg"
        land_path = "" #TODO

        image = Image.open(image_path).convert('RGB')

        # Calculate landmarks
        _landmarks = landmark_inference_func(image)
        lands.append(flatten(_landmarks))

        crop_image = crop_pil_image_face(pil_image=image, landmarks_106=_landmarks)

        crop_image.save(os.path.join(video_feature_folder, crop_image_path))
        img_paths.append([crop_image_path, land_path])


    # Save features in to folder "*/feature/" as AnnVI-like
    # reference: data/AnnScripts/script_AnnVI.py
    ratio = 0.9
    length = len(img_paths)
    length_train = int(length * ratio)
    index = list(range(length))
    random.shuffle(index)
    # train dataset
    img_paths_train = [img_paths[i] for i in index[:length_train]]
    audio_features_train = [audio_features[i] for i in index[:length_train]]
    lands_train = [lands[i] for i in index[:length_train]]
    poses_train = [poses[i] for i in index[:length_train]]
    eyes_train = [eyes[i] for i in index[:length_train]]
    # test dataset
    img_paths_test = [img_paths[i] for i in index[length_train:]]
    audio_features_test = [audio_features[i] for i in index[length_train:]]
    lands_test = [lands[i] for i in index[length_train:]]
    poses_test = [poses[i] for i in index[length_train:]]
    eyes_test = [eyes[i] for i in index[length_train:]]

    print("* Save data")
    save_data_train = {'img_paths': img_paths_train, 'audio_features': audio_features_train, 'lands': lands_train,
                       'poses': poses_train, 'eyes': eyes_train}
    torch.save(save_data_train, os.path.join(video_feature_folder, '{}_train.t7'.format(img_size)))

    save_data_test = {'img_paths': img_paths_test, 'audio_features': audio_features_test, 'lands': lands_test,
                      'poses': poses_test, 'eyes': eyes_test}
    torch.save(save_data_test, os.path.join(video_feature_folder, '{}_test.t7'.format(img_size)))


if __name__ == '__main__':
    data_paths, opt, final_feature_folder = get_video_paths()
    inference_HRNet_func = init_HRNet(opt.rf_model_path, opt.lm_model_path)

    for data_path in data_paths:
        process_video_into_folder_feature(
            data_path,
            landmark_inference_func=inference_HRNet_func,
            win_size=opt.win_size,
            sr=opt.sr, n_mfcc=opt.n_mfcc,
            n_fft=opt.n_fft,
            hop_length=opt.hop_length,
        )

    print(f"To run train*.py, please additionally assign:\n  --data_root {final_feature_folder}")