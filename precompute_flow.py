import sys
sys.path.append('/home/eisneim/www/ml/_video_gen/ltx_training/SEA-RAFT/core')

import moviepy.editor as mpy

import os, random, math, time
import subprocess
import numpy as np
from tqdm import tqdm
import cv2
import torch
from torchvision import transforms
from PIL import Image
import argparse
# import math
import json
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic

from raft import RAFT
from raft_utils.flow_viz import flow_to_image


from precompute import get_frames, VideoFramesDataset

# ckpt_path = "SEA-RAFT/ckpt/Tartan-C-T-TSKH-spring540x960-M.pth"
# cfg_path = "SEA-RAFT/config/eval/spring-M.json"
ckpt_path = "SEA-RAFT/ckpt/Tartan-C-T432x960-M.pth"
cfg_path = "SEA-RAFT/config/eval/sintel-L.json"

def json_to_args(json_path):
    # return a argparse.Namespace object
    with open(json_path, 'r') as f:
        data = json.load(f)
    args = argparse.Namespace()
    args_dict = args.__dict__
    for key, value in data.items():
        args_dict[key] = value
    return args

args = json_to_args(cfg_path)
args.cfg = cfg_path
args.path = ckpt_path
args.device= "cuda"
args.url = None

model_raft = RAFT(args)
state_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))
model_raft.load_state_dict(state_dict, strict=False)
model_raft = model_raft.to(args.device)
model_raft.eval()

class KalmanFilter:
    def __init__(self, initial_state, process_noise, measurement_noise):
        """
        Initialize the Kalman Filter.
        
        Args:
            initial_state (torch.Tensor): Initial state estimate (shape: [H, W, 2]).
            process_noise (float): Process noise covariance.
            measurement_noise (float): Measurement noise covariance.
        """
        self.state = initial_state  # State estimate (u, v)
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        
        # Initialize covariance matrix for the state estimate
        self.covariance = torch.eye(2).unsqueeze(0).unsqueeze(0) * 1.0  # Shape: [1, 1, 2, 2]
        self.covariance = self.covariance.expand(
            initial_state.shape[0], initial_state.shape[1], 2, 2
        ).to(initial_state.device)  # Shape: [H, W, 2, 2]

    def predict(self):
        """
        Predict the next state and covariance.
        """
        # Assume constant velocity model (state does not change without external input)
        self.state = self.state  # No change in state prediction
        pp = torch.eye(2).unsqueeze(0).unsqueeze(0).to(self.state.device)
        self.covariance = self.covariance + pp * self.process_noise

    def update(self, measurement):
        """
        Update the state estimate based on the measurement.
        
        Args:
            measurement (torch.Tensor): Observed optical flow (shape: [H, W, 2]).
        """
        # Compute Kalman gain
        H = torch.eye(2).unsqueeze(0).unsqueeze(0).to(self.state.device)  # Observation matrix (identity for direct observation)
        H = H.expand(measurement.shape[0], measurement.shape[1], 2, 2)  # Shape: [H, W, 2, 2]
        
        S = self.covariance + torch.eye(2).unsqueeze(0).unsqueeze(0).to(self.state.device) * self.measurement_noise  # Innovation covariance
        K = torch.matmul(self.covariance, torch.inverse(S))  # Kalman gain
        
        # Update state estimate
        y = measurement - self.state  # Innovation (residual)
        self.state = self.state + torch.matmul(K, y.unsqueeze(-1)).squeeze(-1)
        
        # Update covariance
        I = torch.eye(2).unsqueeze(0).unsqueeze(0).to(self.state.device)  # Identity matrix
        I = I.expand(measurement.shape[0], measurement.shape[1], 2, 2)  # Shape: [H, W, 2, 2]
        self.covariance = torch.matmul(I - torch.matmul(K, H), self.covariance)

def smooth_optical_flow(optical_flow, process_noise=1e-3, measurement_noise=1e-1):
    """
    Smooth an optical flow sequence using a Kalman filter.
    
    Args:
        optical_flow (torch.Tensor): Optical flow sequence (shape: [T, H, W, 2]).
        process_noise (float): Process noise covariance.
        measurement_noise (float): Measurement noise covariance.
    
    Returns:
        torch.Tensor: Smoothed optical flow sequence (shape: [T, H, W, 2]).
    Weak Smoothing : process_noise=0.1, measurement_noise=0.001
    Moderate Smoothing : process_noise=1e-2, measurement_noise=1e-2
    Strong Smoothing : process_noise=1e-3, measurement_noise=1e-1
     
    """
    T, H, W, _ = optical_flow.shape
    device = optical_flow.device
    smoothed_flow = torch.zeros_like(optical_flow, device=device)
    
    # Initialize Kalman filter with the first frame as the initial state
    kf = KalmanFilter(initial_state=optical_flow[0], process_noise=process_noise, measurement_noise=measurement_noise)
    
    for t in range(T):
        kf.predict()
        kf.update(optical_flow[t])
        smoothed_flow[t] = kf.state
    
    return smoothed_flow


def get_optical_flow(frames):
    flows = []
    # for idx in tqdm(range(len(frames) - 1), "generate flow"):
    for idx in range(len(frames) - 1):
        # ch, h, w
        image1 = torch.tensor(frames[idx], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(args.device)
        image2 = torch.tensor(frames[idx + 1], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(args.device)
        
        with torch.no_grad():
            output = model_raft(image1, image2, iters=args.iters, test_mode=True)
            flow = output['flow'][-1]
            info = output['info'][-1]
            flows.append(flow.permute(0, 2, 3, 1))

    flows = torch.cat(flows, dim=0)
    # print("smothing", flows.shape)
    smothed = smooth_optical_flow(flows, process_noise=0.02, measurement_noise=0.01).cpu().numpy()
    flows_vis = [ flow_to_image(ff, convert_to_bgr=False) for ff in smothed ]

    return flows_vis


class VideoFlowDataset(VideoFramesDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def cache_flows(self, to_latent, device="cuda"):
        print(f"building caches, video count: {len(self.videos)}")
        latent_num_frames = (self.num_frames - 1) / 8 + 1

        for ii, vid in enumerate(tqdm(self.videos)):
            dest = os.path.join(self.cache_dir, os.path.basename(vid).rsplit(".", 1)[0] + "_flow.pt")
            dest_flow_video = dest.replace("_flow.pt", "_flow.mp4")
            ic(dest)
            if os.path.exists(dest):
                # print("skip:", dest)
                continue
            try:
                video_frames = get_frames(vid, self.width, self.height, 0,  f=self.get_frames_max, fps=self.fps)
            except:
                print("error file:", vid)
                continue
            if len(video_frames) < self.num_frames:
                continue
            # remove first frame, why? because frame blending motion blur producec bad first frame
            video_frames = video_frames[1:]

            _, f_height, f_width, f_ch = video_frames.shape
            # ignore vertical video
            if f_height > f_width:
                print("ignore vertical for now:", vid)
                continue

            # divid into parts
            iters = len(video_frames) // self.num_frames
            flow_latents = []
            infos = []
            first_frames = []
            flows_group = []

            for idx in range(iters):
                frames = video_frames[ idx*self.num_frames : (idx + 1) * self.num_frames + 1 ]
                # 因为计算光流图是两帧之间，所以要多加一帧; 50 video frame -> 49 optical flow
                if len(frames) < self.num_frames + 1:
                    continue

                frames_t = self.to_tensor(frames, device=device)
                ic(frames_t.shape)
                ff, _, height, width = to_latent(frames_t[:, 0:1, :, :, :])
                ic(ff.shape)
                assert ff.ndim == 3, "patched latent should have 3 dims"
                first_frames.append(ff)
                infos.append(dict(num_frames=latent_num_frames, height=height, width=width))
                flow_rgb = get_optical_flow(frames)
                flows_group.append(flow_rgb)

                flow_rgb_t = self.to_tensor(np.stack(flow_rgb), device=device)
                latent, num_frames, _, _ = to_latent(flow_rgb_t)
                flow_latents.append(latent)
            
            if len(flow_latents) == 0:
                continue

            flow_latents = torch.cat(flow_latents, dim=0)

            torch.save(dict(flow_latents=flow_latents, 
              first_frames=first_frames,
              meta_info=infos), dest)
            self.data.append(dest)
            # save flow to mp4
            flows = np.concatenate(flows_group)
            clip = mpy.ImageSequenceClip([ ff for ff in flows ], fps=24)
            clip.write_videofile(dest_flow_video, fps=24, logger=None)



if __name__ == "__main__":
    from tqdm import tqdm

    from ltx_video_lora import *

    # ------------------- 
    # prompt_prefix = "freeze time, camera orbit left,"
    video_dir = [
        "/media/eisneim/teli-disk/ltx_train_data/3dgs",
        "/media/eisneim/teli-disk/ltx_train_data/game_p9",
        "/media/eisneim/teli-disk/ltx_train_data/game_p10",
        "/media/eisneim/teli-disk/ltx_train_data/game_p11",
        "/media/eisneim/teli-disk/ltx_train_data/game_p12",
        "/media/eisneim/teli-disk/ltx_train_data/game_p13",
    ]
    cache_dir = "/media/eisneim/4T/ltx_optical_flow/all_video_flow"

    # video_dir = "/media/eisneim/KESU/dataset/splat_screen_rec/game_part6"
    # cache_dir = "/media/eisneim/KESU/dataset/game_part6"

    # video_dir = "/media/eisneim/KESU/dataset/splat_screen_rec"
    # cache_dir = "/media/eisneim/KESU/dataset/orbit_precomputed"

    # video_dir = [
    #     '/media/eisneim/Evie/gid_data/no_watermark_portrait',
    #     '/media/eisneim/Evie/gid_data/pexels_potential', 
    #     '/media/eisneim/Evie/gid_data/pexels_static', 
    #     '/media/eisneim/Evie/gid_data/pexels_static_groupd', 
    #     '/media/eisneim/Evie/gid_data/pexels_temp', 
    #     '/media/eisneim/Evie/gid_data/pexels_temp2', 
    #     '/media/eisneim/Evie/gid_data/_self_filmed',
    # ]
    # cache_dir = "/media/eisneim/teli-disk/ltx_train_data/gid_nowatermark"
    # prompt_prefix = ""

    # cache_dir = "/home/eisneim/www/ml/video_gen/ltx_training/data/ltxv_disney"
    # video_dir = [

    # ]
    
    dtype = torch.bfloat16
    dataset = VideoFlowDataset(
        video_dir=video_dir,
        cache_dir=cache_dir,
        # width=512,
        # height=288,
        # num_frames= 121,
        # ---------------
        # width=640,
        # height=352, # 360
        # num_frames= 121,
        # width=960,
        # height=544,
        # num_frames= 65,
        # ---------------
        # width=1024,
        # height=576,
        # num_frames= 49,
        width=1280,
        height=704,
        num_frames= 57,
    )

    vae = load_latent_models()["vae"].to(args.device, dtype=dtype)


    def to_latent(frames_tensor):
        ic(frames_tensor.shape)
        assert frames_tensor.size(2) == 3, f"frames should be in shape: (b, f, c, h, w) provided: {frames_tensor.shape}"
        with torch.no_grad():
            data = prepare_latents(
                    vae=vae,
                    image_or_video=frames_tensor,
                    device=args.device,
                    dtype=dtype,
                )
        return data["latents"].cpu(), data["num_frames"], data["height"], data["width"]
            
    
    dataset.cache_flows(to_latent)

    print("done!")


