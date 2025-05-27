import os, random, math, time
import subprocess
import numpy as np
from tqdm import tqdm
from pathlib import Path
import cv2
import torch
from torchvision import transforms
from PIL import Image
from diffusers.utils import export_to_video
from diffusers.video_processor import VideoProcessor

from ltx_video.models.autoencoders.vae_encode import (
    get_vae_size_scale_factor,
    latent_to_pixel_coords,
    vae_decode,
    # vae_encode,
)

from dataset import PrecomputedDataset
from ltx_video_lora import *
device = "cuda"
dtype = torch.bfloat16
# ------------------- 

vae = load_latent_models()["vae"].to(device, dtype=dtype)

# def _unpack_latents(
#         latents: torch.Tensor, num_frames: int, height: int, width: int, patch_size: int = 1, patch_size_t: int = 1
#     ) -> torch.Tensor:
#     batch_size = latents.size(0)
#     latents = latents.reshape(batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
#     latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
#     return latents

def _normalize_latents(
    latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0,
    reverse=False,
) -> torch.Tensor:
    # Normalize latents across the channel dimension [B, C, F, H, W]
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    if not reverse:
        latents = (latents - latents_mean) * scaling_factor / latents_std
    else:
        latents = latents * latents_std / scaling_factor + latents_mean
    return latents


dest_dir = "/home/eisneim/www/ml/_video_gen/LTX-Video-3-22/data/images/dataset_extracted"
dataset_dirs = [
    '/media/eisneim/4T/ltx_data_49_blured/game_p11_49x1024x576',
    # '/media/eisneim/4T/ltx_data_49_blured/game_p10_blured_49x1024x576', 
    # '/media/eisneim/4T/ltx_data_49_blured/game_p9_49x1024x576', 
    # '/media/eisneim/4T/ltx_data_49_blured/game_p8_49x1024x576', 
    # '/media/eisneim/4T/ltx_data_49_blured/3dgs_game_1-6_49x1024x576', 
    # '/media/eisneim/4T/ltx_data_49_blured/game_p7_49x1024x576', 
    # "/media/eisneim/4T/ltx_data_81x1024x576/game_p13_81x1024x576",
]


for dirname in dataset_dirs:
    dataset = PrecomputedDataset(dirname)

    timestep = torch.tensor([0.05], device=device, dtype=dtype)
    # timestep = None
    is_video = False


    for idx, data in enumerate(dataset):
        if idx > 2:
            break

        _, first_frame, _, _, caption, info = data
        frame = unpack_latents(first_frame.unsqueeze(0).to(device, dtype=dtype), 1, info["height"],  info["width"])
        frame = _normalize_latents(frame, vae.mean_of_means, vae.std_of_means, reverse=True)
        
        with torch.no_grad():
            # video =  vae.decode(lt, timestep, return_dict=False)[0]
            image = vae_decode(
                frame, vae, is_video,
                vae_per_channel_normalize=False,
                timestep=timestep,
            )
        image = image.squeeze(2)[0].permute(1, 2, 0)
        img_np = ((image + 1)/ 2 * 255).cpu().float().numpy().astype(np.uint8)
        img = Image.fromarray(img_np)

        basename = os.path.join(dest_dir, str(time.time()))

        img.save(basename + ".jpg")

        with open(basename + ".txt", "w") as ff:
            ff.write(caption)


pirnt("done!")