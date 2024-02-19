import os
import torch
from load_utils import load_model
from guided_diffusion import dist_util
from guided_diffusion.gaussian_diffusion import _encode, _decode
from guided_diffusion.pr_datasets_all import load_data
from tqdm import tqdm
from guided_diffusion.midi_util import visualize_full_piano_roll, save_piano_roll_midi
from music_rule_guidance import music_rules
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = (20,3)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


MODEL_NAME = 'kl/f8-all-onset'
MODEL_CKPT = 'taming-transformers/checkpoints/all_onset/epoch_14.ckpt'

TOTAL_BATCH = 256


def main():

    data = load_data(
        data_dir='datasets/all-len-40-gap-16-no-empty_train.csv',
        batch_size=32,
        class_cond=True,
        image_size=1024,
        deterministic=False,
        fs=100,
    )
    embed_model = load_model(MODEL_NAME, MODEL_CKPT)
    del embed_model.loss
    embed_model.to(dist_util.dev())
    embed_model.eval()

    z_list = []
    with torch.no_grad():
        for _ in tqdm(range(TOTAL_BATCH)):
            batch, cond = next(data)
            batch = batch.to(dist_util.dev())
            enc = _encode(batch, embed_model, scale_factor=1.)
            z_list.append(enc.cpu())
    latents = torch.concat(z_list, dim=0)
    scale_factor = 1. / latents.flatten().std().item()
    print(f"scale_factor: {scale_factor}")
    print("done")



if __name__ == "__main__":
    main()
