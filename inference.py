import numpy as np
import cv2
import os
from PIL import Image
from typing import List, Tuple
import math
import librosa
import python_speech_features as psf
import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms

from util.options import get_opt
from model.audio_net import AudioNet
from model.NAS_GAN import NAS_GAN
from util.net_util import init_net, print_networks
from mlcandy.face_detection.valid_face_utils import get_angle_of_image


def init_A(opt, device):
    netA = AudioNet()
    netA.load_state_dict(torch.load(
        'model/pretrained/{}_best_{}.pth'.format(opt.data, opt.img_size),
        map_location={'cuda:0': 'cuda:0'})['audio_net'])
    netA.to(device)
    return netA


def init_G(opt, device):
    layers = 9
    width_mult_list = [4. / 12, 6. / 12, 8. / 12, 10. / 12, 1.]
    width_mult_list_sh = [4 / 12, 6. / 12, 8. / 12, 10. / 12, 1.]
    state = torch.load('model/NAS_GAN_arch.pt', map_location='cpu')
    netG = NAS_GAN(state['alpha'], state['ratio'], state['ratio_sh'], layers=layers,
                        width_mult_list=width_mult_list, width_mult_list_sh=width_mult_list_sh)
    netG = init_net(netG, opt.init_type)
    print_networks(netG)
    if opt.resume:
        checkpoint = torch.load(
            '{}/{}_{}_G.pth'.format(opt.logdir, opt.resume_epoch if opt.resume_epoch > -1 else 'latest', opt.img_size),
            map_location=device)
        netG.load_state_dict(checkpoint['netG'])
    netG.eval()
    return netG


class InferenceDataset(Dataset):
    def __init__(self, opt, ref_video_path: str, apb_vcharactor_name: str):
        self.fps = cv2.VideoCapture(ref_video_path).get(cv2.CAP_PROP_FPS)
        self.n_fft = 2048  # 44100/30 1470
        self.hop_length = 512  # 44100/60 735
        self.n_mfcc = 20
        self.sr = 44100
        self.win_size = 64

        self.transforms_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.all_aud_feat, self.all_pose, self.all_eye = self.extract_features(ref_video_path)
        self.target_img_abs_paths = self.generate_apb_output_images(opt, apb_vcharactor_name)
        self.target_img_abs_paths = self.target_img_abs_paths[:len(self)]

    def extract_pose_eye(self, images) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        poses, eyes = [], []
        for image in images:
            has_face, roll, pitch, yaw = get_angle_of_image(image) #TODO sequence
            if not has_face:
                raise ValueError

            pose = [float(roll)/180, float(pitch)/180, float(yaw)/180]
            eye = [0.3502,0.3707] # average value in testset #TODO eye value not real

            poses.append(torch.tensor(pose))
            eyes.append(torch.tensor(eye))
        return poses, eyes

    def extract_audio_feature(self, audio_path, image_num) -> List[torch.Tensor]:
        sig, rate = librosa.load(audio_path, sr=self.sr, duration=None)
        f_mfcc = librosa.feature.mfcc(sig, rate, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length)
        f_mfcc_delta = psf.base.delta(f_mfcc, 2)
        f_mfcc_delta2 = psf.base.delta(f_mfcc_delta, 2)
        f_mfcc_all = np.concatenate((f_mfcc, f_mfcc_delta, f_mfcc_delta2), axis=0)

        audio_features = []
        for cnt in range(image_num):
            c_count = int(cnt / self.fps * rate / self.hop_length)
            audio_feat = f_mfcc_all[:, c_count - self.win_size // 2: c_count + self.win_size // 2].transpose(1, 0)
            audio_features.append(torch.from_numpy(audio_feat).unsqueeze(dim=0))

        return audio_features

    def split_video(self, video_path):
        pil_images = []
        cap = cv2.VideoCapture(video_path)
        ret, img = cap.read()
        while ret:
            pil_images.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            ret, img = cap.read()
        else:
            print("Wrong")

        audio_path = os.path.splitext(video_path)[0] + ".wav"
        # os.system(f"ffmpeg -i {video_path} -ab 160k -ac 2 -ar 44100  -vn {audio_path}")
        return audio_path, pil_images

    def extract_features(self, video_path):
        audio_path, images = self.split_video(video_path)

        poses, eyes = self.extract_pose_eye(images)
        audio_features = self.extract_audio_feature(audio_path, len(images))

        return audio_features, poses, poses

    def generate_apb_output_images(self, opt, apb_vcharactor_name: str) -> list:
        idt_path = '{}/{}'.format(opt.data_root, apb_vcharactor_name)
        idt_pack = '{}/{}_test.t7'.format(idt_path, opt.img_size)
        idt_files = torch.load(idt_pack)
        img_paths = idt_files['img_paths']

        img_abs_paths = []
        for img_path in img_paths:
            img_abs_paths.append(f"{idt_path}/{img_path[0]}")

        scale = 1.0 * len(self) / len(img_abs_paths)
        if scale > 1:
            scale = int(math.ceil(scale))
            img_abs_paths *= scale

        return img_abs_paths


    def __getitem__(self, index):
        img = Image.open(self.target_img_abs_paths[index]).convert('RGB')
        img = self.transforms_image(img)

        return self.all_aud_feat[index], self.all_pose[index], self.all_eye[index], img

    def __len__(self):
        return len(self.all_aud_feat)


def inference(ref_video_path: str, target_video_path: str, output_video_path: str):
    opt = get_opt()
    opt.data = 'AnnVI'
    opt.data_root = '/usr/stable/apb/raw/AnnVI/feature'
    opt.img_size = 256
    opt.resume = True
    opt.resume_name = 'AnnVI-Big'
    opt.logdir = '{}/{}'.format(opt.checkpoint, opt.resume_name)
    opt.resume_epoch = -1
    opt.gpus = [0]
    opt.results_dir = '{}/results/'.format(opt.logdir)
    opt.video_repeat_times = 5
    opt.aud_counts = 300
    device = torch.device('cuda:{}'.format(opt.gpus[0])) if opt.gpus[0] > -1 else torch.device('cpu')
    apb_vcharactor_name = "man1"

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 25.0, (opt.img_size, opt.img_size))

    netA = init_A(opt, device)
    netG = init_G(opt, device)

    dataset = InferenceDataset(opt, ref_video_path, apb_vcharactor_name)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1)

    for batch_idx, test_data in enumerate(dataloader):
        # feed the input
        aud_feat, pose, eye, real_img = test_data
        aud_feat = aud_feat.to(device)
        pose = pose.to(device)
        eye = eye.to(device)
        real_img = real_img.to(device)

        # forward
        latent, landmark = netA(aud_feat, pose, eye)
        fake_img = netG(real_img, latent)

        # post-processing
        fake_img = fake_img.data[0].cpu().numpy()
        fake_img_numpy = (np.transpose(fake_img, (1, 2, 0)) + 1) / 2.0 * 255.0
        fake_img_numpy = cv2.cvtColor(fake_img_numpy, cv2.COLOR_BGR2RGB)
        fake_img_numpy = fake_img_numpy.astype(np.uint8)
        video_writer.write(fake_img_numpy)


if __name__ == '__main__':
    ref_video_path = "/usr/stable/apb/raw/liza/video/result_18s.mp4"
    target_video_path = ""
    output_video_path = "/tmp/result_18s_apb.mp4"

    inference(ref_video_path, target_video_path, output_video_path)
