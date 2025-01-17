import os
from tqdm import tqdm
from typing import List, Tuple
import numpy as np
import cv2
from PIL import Image
import torch
from moviepy.editor import VideoFileClip
import librosa
import python_speech_features as psf

from mlcandy.face_detection.valid_face_utils import get_angle_of_image


def split_video(video_path, split_output_folder: str) -> Tuple[str, list]:
    pil_image_paths = []
    cap = cv2.VideoCapture(video_path)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    ret, img = cap.read()

    print("  ** Start to split and save frames")
    for cnt in tqdm(range(int(frame_count))):
        if ret:
            pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            pil_image_path = os.path.join(split_output_folder, f"{cnt}.jpg")
            pil_image.save(pil_image_path)
            pil_image_paths.append(pil_image_path)
            ret, img = cap.read()
        else:
            print("Break")
            break

    print("  ** Start to split and save audio")
    audio_path = os.path.splitext(video_path)[0] + ".wav"
    if not os.path.exists(audio_path): # For audio enhancement
        # video = VideoFileClip(video_path)
        # audio = video.audio
        # audio.write_audiofile(audio_path)
        os.system(f"ffmpeg -i {video_path} -ab 160k -ac 2 -ar 44100  -vn {audio_path}")
    return audio_path, pil_image_paths


def extract_pose_eye(image_paths) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    poses, eyes = [], []
    for image_path in tqdm(image_paths):
        image = Image.open(image_path).convert('RGB')
        has_face, roll, pitch, yaw = get_angle_of_image(image) #TODO sequence
        if not has_face:
            raise ValueError

        pose = [float(roll)/180, float(pitch)/180, float(yaw)/180]
        eye = [0.3502, 0.3707] # average value in testset #TODO eye value not real

        poses.append(torch.tensor(pose))
        eyes.append(torch.tensor(eye))
    return poses, eyes


def extract_audio_feature(audio_path, image_num, fps, win_size, sr, n_mfcc, n_fft, hop_length) -> List[torch.Tensor]:
    sig, rate = librosa.load(audio_path, sr=sr, duration=None)
    f_mfcc = librosa.feature.mfcc(sig, rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    f_mfcc_delta = psf.base.delta(f_mfcc, 2)
    f_mfcc_delta2 = psf.base.delta(f_mfcc_delta, 2)
    f_mfcc_all = np.concatenate((f_mfcc, f_mfcc_delta, f_mfcc_delta2), axis=0)

    audio_features = []
    for cnt in tqdm(range(image_num)):
        c_count = int(cnt / fps * rate / hop_length)
        start_index = c_count - win_size // 2
        end_index = c_count + win_size // 2

        # Reconstruct by myself
        if start_index <0:
            start_index =0
            repeat_pattern = [win_size -end_index ] +[1 ] *(f_mfcc_all.shape[1 ] -1)
            pad_f_mfcc_all = np.repeat(f_mfcc_all, repeats=repeat_pattern, axis=1) # Padding with the first column
            audio_feat = pad_f_mfcc_all[:, start_index: start_index +win_size].transpose(1, 0)
        elif end_index >f_mfcc_all.shape[1]:
            repeat_pattern = [1 ] *(f_mfcc_all.shape[1 ] -1) + [win_size +start_index -f_mfcc_all.shape[1] +1]
            pad_f_mfcc_all = np.repeat(f_mfcc_all, repeats=repeat_pattern, axis=1) # Padding with the first column

            audio_feat = pad_f_mfcc_all[:, start_index: start_index + win_size].transpose(1, 0)
        else:
            audio_feat = f_mfcc_all[:, start_index: end_index].transpose(1, 0)

        audio_feat = torch.from_numpy(audio_feat).unsqueeze(dim=0)

        audio_features.append(audio_feat)

    return audio_features


def extract_features(audio_path, image_paths: List[str], fps, win_size, sr, n_mfcc, n_fft, hop_length) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    print("  ** Start extract pose and eye")
    poses, eyes = extract_pose_eye(image_paths)

    print("  ** Start extract audio feature")
    audio_features = extract_audio_feature(
        audio_path, len(image_paths),
        fps=fps,
        win_size=win_size,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length)

    return audio_features, poses, eyes
