import librosa

def get_w2l_mel(audio_path):
    pass

def get_apb_mfcc(audio_path):
    sig, rate = librosa.load(audio_path, sr=sr, duration=None)
    time_duration = len(sig) / rate
    # print('time of duration : {}'.format(time_duration))l
    f_mfcc = librosa.feature.mfcc(sig, rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    f_mfcc_delta = psf.base.delta(f_mfcc, 2)
    f_mfcc_delta2 = psf.base.delta(f_mfcc_delta, 2)
    f_mfcc_all = np.concatenate((f_mfcc, f_mfcc_delta, f_mfcc_delta2), axis=0)

    return f_mfcc_all


if __name__ == '__main__':
    audio_path = ""

    get_apb_mfcc(audio_path)
    get_w2l_mel(audio_path)
