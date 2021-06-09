import argparse
import numpy as np
import cv2
from PIL import Image
import math
import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from datav2.common import extract_features

from util.options import get_opt
from model.audio_net import AudioNet
from model.NAS_GAN import NAS_GAN
from util.net_util import init_net, print_networks


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

        self.all_aud_feat, self.all_pose, self.all_eye = extract_features(
            ref_video_path,
            fps=self.fps,
            win_size=self.win_size,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        self.target_img_abs_paths = self.generate_apb_output_images(opt, apb_vcharactor_name)
        self.target_img_abs_paths = self.target_img_abs_paths[:len(self)]

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
        # print(aud_feat.shape, pose.shape, eye.shape)
        latent, landmark = netA(aud_feat, pose, eye)
        fake_img = netG(real_img, latent)

        # post-processing
        fake_img = fake_img.data[0].cpu().numpy()
        fake_img_numpy = (np.transpose(fake_img, (1, 2, 0)) + 1) / 2.0 * 255.0
        fake_img_numpy = cv2.cvtColor(fake_img_numpy, cv2.COLOR_BGR2RGB)
        fake_img_numpy = fake_img_numpy.astype(np.uint8)
        video_writer.write(fake_img_numpy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_video_path", "-r", dest="ref_video_path", default="/usr/stable/apb/raw/liza/video/c230.mp4")
    parser.add_argument("--output_video_path", "-o", dest="output_video_path", default="/tmp/c230_apb.mp4")

    opt = parser.parse_args()
    target_video_path = ""

    inference(opt.ref_video_path, target_video_path, opt.output_video_path)
