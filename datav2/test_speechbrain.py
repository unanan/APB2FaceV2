import torch
import argparse
import torchaudio
from speechbrain.pretrained import SpectralMaskEnhancement


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", default="output.wav")
    opt = parser.parse_args()

    enhance_model = SpectralMaskEnhancement.from_hparams(
        source="speechbrain/metricgan-plus-voicebank",
        savedir="pretrained_models/metricgan-plus-voicebank",
        run_opts={"device":"cuda"},
    )

    # Load and add fake batch dimension
    noisy = enhance_model.load_audio(opt.input).unsqueeze(0)

    # Add relative length tensor
    enhanced = enhance_model.enhance_batch(noisy, lengths=torch.tensor([1.]))

    # Saving enhanced signal on disk
    torchaudio.save(opt.output, enhanced.cpu(), 16000)