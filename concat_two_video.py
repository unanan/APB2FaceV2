import cv2
import argparse
from tqdm import tqdm
import numpy as np
import os

def get_options():
    parser = argparse.ArgumentParser(description="Get specific options information in a command")
    parser.add_argument("--left_video_path", "-l", dest="left_video_path", type=str, default="", help="Video path show on the left")
    parser.add_argument("--right_video_path", "-r", dest="right_video_path", type=str, default="", help="Video path show on the right")
    parser.add_argument("--audio_path", "-a", dest="audio_path", type=str, default="", help="Audio path")
    parser.add_argument("--output_video_path", "-o", dest="output_video_path", type=str, default="", help="Output video path")

    return parser.parse_args()


def write_video_frames(vframes_list: list, output_path):
    # Copy from w2l 3d
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    first_frame = vframes_list[0]
    h,w = first_frame.shape[:2]

    out1 = cv2.VideoWriter(output_path, fourcc, 25, (w, h), True)

    for frame in tqdm(vframes_list):
        out1.write(frame)
    out1.release()


def main():
    opt = get_options()

    cap_left = cv2.VideoCapture(opt.left_video_path)
    cap_right = cv2.VideoCapture(opt.right_video_path)
    ret_left, left_frame = cap_left.read()
    ret_right, right_frame = cap_right.read()

    concat_frames = []
    while ret_left and ret_right:
        #Take the smaller one's size
        if left_frame.shape[0]*left_frame.shape[1]< right_frame.shape[0]*right_frame.shape[1]:
            rsize = (int(right_frame.shape[1]*left_frame.shape[0]/right_frame.shape[0]), left_frame.shape[0])
            right_frame = cv2.resize(right_frame, rsize)
        else:
            rsize = (int(left_frame.shape[1]*right_frame.shape[0]/left_frame.shape[0]), right_frame.shape[0])
            left_frame = cv2.resize(left_frame, rsize)
        concat_frames.append(np.concatenate([left_frame, right_frame], axis=1))
        ret_left, left_frame = cap_left.read()
        ret_right, right_frame = cap_right.read()
    write_video_frames(concat_frames, "tmp.mp4")

    # Extract audio from left video #TODO
    if not opt.audio_path:
        audio_path = os.path.splitext(opt.left_video_path)[0]+".wav"
        command = f"ffmpeg -i {opt.left_video_path} -vn -acodec copy {audio_path}"
        command = f"ffmpeg -i {opt.left_video_path} -q:a 0 -map a {audio_path}"
        os.system(command)
    else:
        audio_path = opt.audio_path

    # Write audio to concatenated video
    command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(audio_path, "tmp.mp4", opt.output_video_path)
    os.system(command)
    print(f"Result video write to: {opt.output_video_path}")


if __name__ == '__main__':
    main()