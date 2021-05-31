from util.options import get_opt
from trainer.Demo_l2face_trainer import Trainer_
import torch
from data.Demo_ann import Dataset_

def init_A(data, img_size):
    netA = AudioNet()
    netA.load_state_dict(torch.load('model/pretrained/{}_best_{}.pth'.format(data, img_size),
                                         map_location={'cuda:0': 'cuda:0'})['audio_net'])
    netA.to(self.device)
    return netA


def init_G():
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
            map_location=self.device)
        netG.load_state_dict(checkpoint['netG'])
    netG.eval()
    return netG


def write_video():
    pass

def inference(audio_path, output_video_path):
    opt = get_opt()
    opt.data = 'AnnVI'
    opt.data_root = '/usr/stable/apb/raw/AnnVI/feature'
    opt.img_size = 256
    opt.resume = True
    opt.resume_name = 'AnnVI-Big'
    opt.logdir = '{}/{}'.format(opt.checkpoint, opt.resume_name)
    opt.resume_epoch = -1
    opt.gpus = [0]
    opt.results_dir = '{}/results/{}'.format(opt.logdir, mode)
    opt.video_repeat_times = 5
    opt.aud_counts = 300

    netA = init_A(data, img_size, gpus)
    netG = init_G()


    for batch_idx, test_data in enumerate(dataloader):
        aud_feat1 = torch.tensor(aud_feat1).unsqueeze(dim=0)
        aud_feat1.to(self.device)


        latent, landmark = netA(self.aud_feat1, self.pose1, self.eye1)
        self.img1_fake = netG(self.img2, latent)
        img1_fake = self.img1_fake.data[0].cpu().numpy()
        img1_fake_numpy = (np.transpose(img1_fake, (1, 2, 0)) + 1) / 2.0 * 255.0
        img1_fake_numpy = cv2.cvtColor(img1_fake_numpy, cv2.COLOR_BGR2RGB)
        img1_fake_numpy = img1_fake_numpy.astype(np.uint8)


if __name__ == '__main__':
    audio_path = "/tmp/A0008_1.wav"
    output_video_path = "/tmp/A0008_1_apb.mp4"

    inference(audio_path, output_video_path)
